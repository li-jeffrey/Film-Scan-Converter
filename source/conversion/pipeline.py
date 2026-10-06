"""Negative conversion pipeline: grid -> analysis -> normalize -> render.

Stage order follows NegPy's NormalizationProcessor / PhotometricProcessor /
TransparencyBaseProcessor / TransferProcessor (GPL-3.0), minus the dropped features listed
in docs/negpy-port-design.md.

Linear input is float32 RGB in [0, 1] (uint16 is accepted and scaled). Rendering returns
uint16 in the requested output space: (H, W, 3) RGB, or (H, W) for B&W.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

from . import crosstalk as xt
from .constants import EXPOSURE_CONSTANTS, TRANSFER_DENSITY_RANGE
from .density import (
    LogNegativeBounds,
    analyze_log_exposure_bounds_from_log,
    luminance_density_range,
    measure_anchor_from_log,
    measure_highlight_point_from_log,
    measure_neutral_axis_from_log,
    measure_shadow_point_from_log,
    measure_shadow_refs_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
    prefilter_log_grid,
    sorted_channel_grid,
    to_log_density,
    unmix_log_image,
)
from .modes import ProcessMode, per_channel_point_offsets
from .output import encode_display, luminance, to_uint
from .papers import effective_paper_profile
from .params import ConversionParams
from .print_curve import (
    apply_characteristic_curve,
    cast_solve_inputs,
    filtration_offsets,
    grade_coupled_shape,
    highlight_hold_offset,
    neutral_axis_affine,
    per_channel_curve_params,
    slope_to_grade,
)
from .transfer import (
    apply_camera_matrix,
    apply_transfer_curve,
    camera_to_working_matrix,
    transfer_assumed_anchor,
    transfer_auto_terms,
    transfer_bounds,
    transfer_curve_params,
    transfer_point_offsets,
    transfer_widths,
)

CHUNK_PIXELS = 1 << 20


def as_linear_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return img.astype(np.float32) * np.float32(1.0 / 65535.0)
    return np.asarray(img, dtype=np.float32)


@dataclass
class Grid:
    """Block-median log grid (post camera matrix and unmix) and its per-channel sort."""

    log: np.ndarray
    sorted: np.ndarray
    cam_matrix: Optional[np.ndarray] = None
    unmix: Optional[np.ndarray] = None


@dataclass
class Analysis:
    """Everything the meters produce. Small and picklable (shipped to export workers)."""

    bounds: LogNegativeBounds  # pre-trim, from the luma/colour clips or the fixed slide window
    shadow_refs: Optional[tuple] = None
    neutral_axis: Optional[tuple] = None
    anchor: Optional[float] = None
    textural_range: Optional[float] = None
    shadow_point: Optional[float] = None
    highlight_point: Optional[float] = None
    norm_range: Optional[float] = None

    def trimmed_bounds(self, params: ConversionParams) -> LogNegativeBounds:
        wp3, bp3 = per_channel_point_offsets(params.white_point_offset, params.black_point_offset)
        if params.mode == ProcessMode.E6:
            wp3, bp3 = transfer_point_offsets(wp3, bp3)
            if not any(wp3 + bp3):
                return self.bounds
            return LogNegativeBounds(
                floors=tuple(f + w for f, w in zip(self.bounds.floors, wp3)),
                ceils=tuple(c + b for c, b in zip(self.bounds.ceils, bp3)),
            )
        if not any(wp3 + bp3):
            return self.bounds
        return LogNegativeBounds(
            floors=tuple(f + w for f, w in zip(self.bounds.floors, wp3)),
            ceils=tuple(c + b for c, b in zip(self.bounds.ceils, bp3)),
        )


def _unmix_matrix(params: ConversionParams) -> Optional[np.ndarray]:
    mode = params.mode
    if mode != ProcessMode.C41:
        return None
    matrix = xt.get_matrix(params.crosstalk_profile)
    if matrix is None:
        matrix = xt.DEFAULT_CROSSTALK_MATRIX
    return xt.effective_crosstalk_matrix(
        params.crosstalk_strength, tuple(matrix), xt.get_process(params.crosstalk_profile), mode
    )


def camera_matrix(params: ConversionParams, cam_xyz, camera_wb) -> Optional[np.ndarray]:
    """Slides only: camera -> Adobe working space with the as-shot WB folded back in."""
    if params.mode != ProcessMode.E6:
        return None
    return camera_to_working_matrix(cam_xyz, camera_wb)


def build_grid(lin: np.ndarray, params: ConversionParams, cam_matrix: Optional[np.ndarray] = None) -> Grid:
    """lin: the analysis region (already cropped, border excluded) as linear RGB."""
    img = as_linear_float(lin)
    if cam_matrix is not None:
        img = apply_camera_matrix(np.nan_to_num(img, nan=1e-6, posinf=1.0, neginf=1e-6), cam_matrix)
    # The analysis border is applied by the caller's crop (RawProcessing.crop include_EQ_ignore).
    log = prefilter_log_grid(img, None, 0.0)
    um = _unmix_matrix(params)
    log = unmix_log_image(log, um)
    return Grid(log=log, sorted=sorted_channel_grid(log), cam_matrix=cam_matrix, unmix=um)


def analyze(grid: Grid, params: ConversionParams) -> Analysis:
    g = grid.log
    mode = params.mode
    if mode == ProcessMode.E6:
        floors, ceils = transfer_bounds()
        bounds = LogNegativeBounds(floors=floors, ceils=ceils)
        a = Analysis(bounds=bounds, norm_range=luminance_density_range(bounds))
        if params.cast_removal > 0.0:
            a.neutral_axis = measure_neutral_axis_from_log(g, bounds, None, 0.0)
        a.anchor = measure_anchor_from_log(g, bounds, None, 0.0, assumed=transfer_assumed_anchor())
        a.textural_range = measure_textural_range_from_log(g, None, 0.0)
        a.shadow_point = measure_shadow_point_from_log(g, bounds, None, 0.0)
        a.highlight_point = measure_highlight_point_from_log(g, bounds, None, 0.0)
        return a

    override = None
    if params.base_rgb is not None and mode == ProcessMode.C41:
        base = np.clip(np.asarray(params.base_rgb, dtype=np.float32), 1e-6, None).reshape(1, 1, 3)
        base_log = unmix_log_image(np.log10(base), grid.unmix)
        override = tuple(float(v) for v in base_log.reshape(3))
    bounds = analyze_log_exposure_bounds_from_log(
        g, None, 0.0, params.luma_clip, params.color_clip, sorted_grid=grid.sorted, thin_end_override=override
    )
    a = Analysis(bounds=bounds, norm_range=luminance_density_range(bounds))
    if mode == ProcessMode.C41:
        a.shadow_refs = measure_shadow_refs_from_log(g, None, 0.0)
        a.neutral_axis = measure_neutral_axis_from_log(g, bounds, None, 0.0)
    a.anchor = measure_anchor_from_log(g, bounds, None, 0.0)
    a.textural_range = measure_textural_range_from_log(g, None, 0.0)
    a.shadow_point = measure_shadow_point_from_log(g, bounds, None, 0.0)
    a.highlight_point = measure_highlight_point_from_log(g, bounds, None, 0.0)
    return a


def normalize(lin: np.ndarray, params: ConversionParams, analysis: Analysis,
              cam_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Linear RGB -> normalized log density (float32 H,W,3)."""
    img = as_linear_float(lin)
    if cam_matrix is not None:
        img = apply_camera_matrix(np.nan_to_num(img, nan=1e-6, posinf=1.0, neginf=1e-6), cam_matrix)
    log = to_log_density(img)
    log = unmix_log_image(log, _unmix_matrix(params))
    return normalize_log_image(log, analysis.trimmed_bounds(params))


# ---------------------------------------------------------------- curve setup


@dataclass
class PrintSetup:
    slopes: tuple
    pivots: tuple
    curvs: tuple
    toe: float
    shoulder: float
    cmy: tuple
    highlight_density: float
    paper: object
    d_min: float
    effective_grade: Optional[float]


def print_setup(params: ConversionParams, a: Analysis) -> PrintSetup:
    mode = params.mode
    paper = effective_paper_profile(params.paper_profile, mode)
    d_min = paper.d_min if params.paper_dmin else 0.0
    bounds = a.trimmed_bounds(params)
    strength, shadow_norm, neutral_norm = cast_solve_inputs(
        bounds, a.shadow_refs, a.neutral_axis, params.cast_removal if mode != ProcessMode.BW else 0.0
    )
    slopes, pivots, curvs = per_channel_curve_params(
        params.grade, params.density, params.auto_grade, strength, a.norm_range, shadow_norm,
        a.textural_range, d_min=d_min, anchor=a.anchor if params.auto_density else None, paper=paper,
        neutral_axis_norm=neutral_norm, shadow_point=a.shadow_point if params.auto_grade else None,
    )
    hl_auto = 0.0
    if params.auto_grade and a.highlight_point is not None:
        hl_auto = highlight_hold_offset(slopes[1], pivots[1], a.highlight_point, d_min=d_min, paper=paper)
    toe, shoulder = grade_coupled_shape(slopes[1], params.toe, params.shoulder)
    cmy = filtration_offsets((params.wb_cyan, params.wb_magenta, params.wb_yellow), bounds)
    return PrintSetup(slopes, pivots, curvs, toe, shoulder, cmy, params.highlight_density + hl_auto,
                      paper, d_min, slope_to_grade(slopes[1], a.norm_range))


def effective_grade(params: ConversionParams, a: Analysis) -> Optional[float]:
    """Read-out for the 'Effective grade R…' label (None on the slide path)."""
    if params.mode == ProcessMode.E6:
        return None
    return print_setup(params, a).effective_grade


def _slide_config(params: ConversionParams):
    return SimpleNamespace(
        density=params.density, grade=params.grade, toe=params.toe, shoulder=params.shoulder,
        toe_width=params.toe_width, shoulder_width=params.shoulder_width,
        auto_exposure=params.auto_density, auto_normalize_contrast=params.auto_grade,
    )


class _Renderer:
    def __init__(self, params: ConversionParams, a: Analysis):
        self.p, self.a = params, a
        self.mode = params.mode
        if self.mode == ProcessMode.E6:
            cfg = _slide_config(params)
            off, con, toe3, sh3 = transfer_curve_params(cfg)
            off, con, hl_auto = transfer_auto_terms(
                cfg, off, con, a.textural_range, a.anchor, a.shadow_point, a.highlight_point)
            bounds = a.trimmed_bounds(params)
            cmy = filtration_offsets((params.wb_cyan, params.wb_magenta, params.wb_yellow), bounds)
            strength, _s, neutral = cast_solve_inputs(bounds, None, a.neutral_axis, params.cast_removal)
            gain, off_n = neutral_axis_affine(neutral, strength)
            tw3, sw3 = transfer_widths(cfg)
            self.slide = dict(
                exposure_offset=off, contrast=con, toe=toe3, shoulder=sh3, cmy_offsets=cmy,
                toe_widths=tw3, shoulder_widths=sw3, shadow_density=params.shadow_density,
                highlight_density=params.highlight_density + hl_auto, cast_gain=gain,
                cast_offset=tuple(o * TRANSFER_DENSITY_RANGE for o in off_n),
                separation=params.dye_separation,
            )
        else:
            self.ps = print_setup(params, a)

    def curve(self, norm: np.ndarray) -> np.ndarray:
        """Normalized density chunk -> linear reflectance (3ch) or 2-D gray (B&W)."""
        p = self.p
        if self.mode == ProcessMode.E6:
            s = self.slide
            return apply_transfer_curve(
                norm, s["exposure_offset"], s["contrast"], s["toe"], s["shoulder"], s["cmy_offsets"],
                s["toe_widths"], s["shoulder_widths"], shadow_density=s["shadow_density"],
                highlight_density=s["highlight_density"], cast_gain=s["cast_gain"],
                cast_offset=s["cast_offset"], separation=s["separation"])
        ps = self.ps
        if self.mode == ProcessMode.BW:
            norm = luminance(norm)  # panchromatic: one density before the curve
        out = apply_characteristic_curve(
            norm, (ps.pivots[0], ps.slopes[0]), (ps.pivots[1], ps.slopes[1]), (ps.pivots[2], ps.slopes[2]),
            toe=ps.toe, toe_width=p.toe_width, shoulder=ps.shoulder, shoulder_width=p.shoulder_width,
            cmy_offsets=ps.cmy, d_min=ps.d_min, curvatures=ps.curvs, paper=ps.paper,
            bpc=not p.paper_black, shadow_density=p.shadow_density, highlight_density=ps.highlight_density,
            dye_separation=1.0 if self.mode == ProcessMode.BW else p.dye_separation)
        return out


def render(src: np.ndarray, params: ConversionParams, analysis: Analysis, out_space: str = "sRGB",
           norm: Optional[np.ndarray] = None, cam_matrix: Optional[np.ndarray] = None,
           bgr: bool = False, chunk_pixels: int = CHUNK_PIXELS) -> np.ndarray:
    """Render to uint16 in `out_space`. Give `src` (linear RGB) or a cached `norm`.

    Row-chunked: only ~chunk_pixels float32 temporaries exist at once; the result is written
    straight into a preallocated uint16 buffer. `bgr` reverses channels at the boundary.
    B&W renders to a 2-D array in the Gray space regardless of `out_space`.
    """
    base = norm if norm is not None else src
    h, w = base.shape[:2]
    bw = params.mode == ProcessMode.BW
    space = "Gray" if bw else out_space
    out = np.empty((h, w) if bw else (h, w, 3), dtype=np.uint16)
    r = _Renderer(params, analysis)
    rows = max(1, chunk_pixels // max(1, w))
    for y in range(0, h, rows):
        blk = base[y:y + rows]
        n = blk if norm is not None else normalize(blk, params, analysis, cam_matrix)
        lin = r.curve(np.ascontiguousarray(n, dtype=np.float32))
        code = encode_display(lin, space)
        if not bw and bgr:
            code = code[..., ::-1]
        to_uint(code, 16, out=out[y:y + rows])
    return out
