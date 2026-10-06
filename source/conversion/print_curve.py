"""Print-curve engine ported from NegPy features/exposure/logic.py (GPL-3.0).

Vectorized NumPy (no numba). Dropped vs. NegPy: local EV/grade maps, split grade,
regional CMY, preflash, separation damping, per-layer trims. See NOTICE.md.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import EXPOSURE_CONSTANTS
from .modes import ProcessMode, auto_meter_for_mode, cast_removal_for_mode, per_channel_point_offsets  # noqa: F401
from .papers import (
    PaperProfile,
    compose_density_matrices,
    effective_constants,
    resolve_dye_matrix,
    resolve_saturation_matrix,
)

LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722

ImageBuffer = np.ndarray


def ensure_image(a: Any) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _expit(x: Any) -> Any:
    """Numpy implementation of the logistic sigmoid function (scipy.special.expit fallback).

    expit(x) = exp(-logaddexp(0, -x)) — exact and overflow-free for any x.
    """
    return np.exp(-np.logaddexp(0.0, -x))



def _inv_softplus_np(y: Any) -> Any:
    """Inverse of softplus: log(exp(y) - 1), stable for y > 0 (pivot solve)."""
    return np.where(y > 20.0, y, np.log(np.expm1(np.maximum(y, 1e-12))))



def _sigmoid_scalar(x: float) -> float:
    return float(np.exp(-np.logaddexp(0.0, -x)))


def quadratic_core(k: float, pivot: float, curv: float, u: Any) -> Any:
    """k*(u - pivot) + curv*u^2, with u held at the parabola's vertex where it would fold back."""
    if curv != 0.0:
        vertex = -k / (2.0 * curv)
        u = np.maximum(u, vertex) if curv > 0.0 else np.minimum(u, vertex)
    return k * (u - pivot) + curv * u * u


class CharacteristicCurve:
    """
    Asymmetric H&D print curve (toe-linear-shoulder) in density space — the NumPy
    mirror of _apply_print_curve_kernel, used by the curve chart so the displayed
    curve matches the render. Returns density (pre-transmittance/encode). Neutral
    (no regional CMY color), since the chart shows the achromatic transfer; the
    achromatic zone density offsets (shadow/highlight ΔD) are included.
    """

    def __init__(
        self,
        contrast: float,
        pivot: float,
        d_min: float = 0.0,
        toe: float = 0.0,
        toe_width: float = 2.5,
        shoulder: float = 0.0,
        shoulder_width: float = 2.5,
        paper: Optional[PaperProfile] = None,
        midtone_gamma: Optional[float] = None,
        bpc: bool = False,
        shadow_density: float = 0.0,
        highlight_density: float = 0.0,
        curvature: float = 0.0,
    ):
        c = effective_constants(paper)
        ts = float(c["toe_shoulder_strength"])
        self.k = float(contrast)
        self.x0 = float(pivot)
        self.curvature = float(curvature)
        self.d_min = float(d_min)
        self.v_star = _reference_linear_value(d_min, paper)
        self.midtone_gamma = float(c["paper_midtone_gamma"]) if midtone_gamma is None else float(midtone_gamma)
        self.gamma_width = float(c["paper_gamma_width"])
        self.zone_sh_center = float(c["anchor_target_density"]) + float(c["zone_density_shadow_offset"])
        self.zone_hi_center = float(c["anchor_target_density"]) + float(c["zone_density_highlight_offset"])
        self.zone_k = float(c["zone_density_sharpness"])
        self.shadow_density = float(shadow_density)
        self.highlight_density = float(highlight_density)
        self.d_max = float(c["d_max"])
        # The BPC reference mirrors the kernel prologue: achromatic, d_min for the tint.
        self.bpc = bool(bpc)
        db = self.d_max
        if toe * ts < 0.0:
            db = self.d_max + toe * ts * float(c["toe_height"])
        self.bpc_black = 10.0**-db
        wr = float(c["toeshoulder_width_ref"])
        # toe = the shadow (upper) bound, shoulder = the highlight (lower) bound.
        self.a_hl = float(c["shoulder_sharpness_base"]) * wr / max(shoulder_width, 1e-6)
        a_sh_base = float(c["toe_sharpness_base"]) * wr / max(toe_width, 1e-6)
        self.d_min_eff = max(0.0, self.d_min + shoulder * ts * float(c["shoulder_height"]))
        toe_eff = toe * ts
        if toe_eff >= 0.0:
            self.d_max_eff = self.d_max - toe_eff * float(c["toe_height"])
            self.a_sh = a_sh_base
        else:
            self.d_max_eff = self.d_max
            self.a_sh = a_sh_base * (1.0 - toe_eff * 4.0)
        if self.d_max_eff < self.d_min_eff + 0.1:
            self.d_max_eff = self.d_min_eff + 0.1

    def __call__(self, x: ImageBuffer) -> ImageBuffer:
        xv = np.asarray(x, dtype=np.float64)
        if self.curvature != 0.0:
            vertex = -self.k / (2.0 * self.curvature)
            xv = np.maximum(xv, vertex) if self.curvature > 0.0 else np.minimum(xv, vertex)
        v = self.k * (xv - self.x0) + self.curvature * xv * xv
        if self.midtone_gamma != 0.0:
            v = v + self.midtone_gamma * self.gamma_width * np.tanh((v - self.v_star) / self.gamma_width)
        if self.shadow_density != 0.0 or self.highlight_density != 0.0:
            w_zsh = _expit(self.zone_k * (v - self.zone_sh_center))
            w_zhi = 1.0 - _expit(self.zone_k * (v - self.zone_hi_center))
            v = v + self.shadow_density * w_zsh + self.highlight_density * w_zhi
        v1 = self.d_min_eff + np.logaddexp(0.0, self.a_hl * (v - self.d_min_eff)) / self.a_hl
        res = self.d_max_eff - np.logaddexp(0.0, self.a_sh * (self.d_max_eff - v1)) / self.a_sh

        if self.bpc:
            t = 10.0 ** (-res)
            t = (t - self.bpc_black) / (1.0 - self.bpc_black)
            res = -np.log10(np.maximum(t, 1e-12))

        return ensure_image(res)



def per_channel_toe_shoulder(
    toe_eff: float,
    shoulder_eff: float,
    toe_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shoulder_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-layer effective toe/shoulder: grade-coupled global value + per-layer trim,
    clamped to the slider domain. Single source of truth for CPU / GPU / chart.
    """

    def _clamp(v: float) -> float:
        return min(max(v, -1.0), 1.0)

    toe3 = (_clamp(toe_eff + toe_trims[0]), _clamp(toe_eff + toe_trims[1]), _clamp(toe_eff + toe_trims[2]))
    sh3 = (
        _clamp(shoulder_eff + shoulder_trims[0]),
        _clamp(shoulder_eff + shoulder_trims[1]),
        _clamp(shoulder_eff + shoulder_trims[2]),
    )
    return toe3, sh3


def per_channel_widths(
    toe_width: float,
    shoulder_width: float,
    toe_width_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shoulder_width_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-layer effective toe/shoulder widths: global width + per-layer trim,
    clamped to the width slider domain. Single source of truth for CPU / GPU / chart.
    """

    def _clamp(v: float) -> float:
        return min(max(v, 0.1), 5.0)

    tw3 = (_clamp(toe_width + toe_width_trims[0]), _clamp(toe_width + toe_width_trims[1]), _clamp(toe_width + toe_width_trims[2]))
    sw3 = (
        _clamp(shoulder_width + shoulder_width_trims[0]),
        _clamp(shoulder_width + shoulder_width_trims[1]),
        _clamp(shoulder_width + shoulder_width_trims[2]),
    )
    return tw3, sw3


def paper_dmin_rgb(d_min: float, paper: Optional[PaperProfile]) -> Tuple[float, float, float]:
    """
    Per-channel paper-white floor: d_min plus the paper's base tint (a minimum
    dye density — tints highlights, fades toward d_max). All-zero when d_min is 0.
    """
    if d_min <= 0.0 or paper is None:
        base = max(d_min, 0.0)
        return (base, base, base)
    t = paper.base_tint_cmy
    return (max(d_min + t[0], 0.0), max(d_min + t[1], 0.0), max(d_min + t[2], 0.0))



def default_grade_range() -> float:
    """Fallback density range when none is measured: a normal negative's, scaled by K."""
    c = EXPOSURE_CONSTANTS
    return float(c["auto_grade_target"]) * float(c["auto_grade_nominal_ratio"]) * float(c["auto_grade_nominal_range"])


def grade_to_slope(grade: float, density_range: Optional[float]) -> float:
    """
    Straight-line slope k from the grade given as an ISO R paper exposure range
    (R180 very soft ... R50 very hard; R110 ~ classic grade 2 paper). k is the
    literal H&D gamma: contrast = negative density range / paper exposure range,
    like real graded paper — k = grade_contrast_scale * range / (R/100).
    """
    c = EXPOSURE_CONSTANTS
    rng_in = default_grade_range() if density_range is None else density_range
    er = min(max(grade, c["iso_r_min"]), c["iso_r_max"]) / 100.0
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    k = float(c["grade_contrast_scale"]) * rng / er
    return float(min(max(k, c["slope_min"]), c["slope_max"]))


def per_channel_dye_separation(
    dye_separation: float,
    trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """
    Per-layer effective dye-separation k: global value + per-layer trim,
    clamped to a sane matrix-coefficient range. Mirrors
    per_channel_toe_shoulder's global+trim convention.
    """

    def _clamp(v: float) -> float:
        return min(max(v, 0.0), 3.0)

    return (
        _clamp(dye_separation + trims[0]),
        _clamp(dye_separation + trims[1]),
        _clamp(dye_separation + trims[2]),
    )


def slope_to_grade(slope: float, density_range: Optional[float]) -> float:
    """
    Inverse of grade_to_slope: the ISO R paper grade equivalent to an effective
    slope, given the density range that produced it. Used to display the contrast
    the conversion is actually applying (including Auto Grade), on the same ISO R
    scale as the Grade slider. Clamped to the slider's R range.
    """
    c = EXPOSURE_CONSTANTS
    rng_in = default_grade_range() if density_range is None else density_range
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    if slope <= 0:
        return float(c["iso_r_max"])
    er = float(c["grade_contrast_scale"]) * rng / float(slope)
    return float(min(max(er * 100.0, c["iso_r_min"]), c["iso_r_max"]))


def effective_midtone_gamma(paper: Optional[PaperProfile], trim: float) -> float:
    """
    Paper's variable midtone gamma plus the user's additive trim. Single source
    of truth for CPU / GPU / chart.
    """
    return float(effective_constants(paper)["paper_midtone_gamma"]) + float(trim)


def per_channel_midtone_gamma(
    paper: Optional[PaperProfile],
    trim: float,
    snap_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """
    Per-layer effective midtone gamma: paper baseline + global trim + Snap trim.
    Single source of truth for CPU / GPU / chart. Unclamped: the slider domains
    keep |gamma| < 1, the kernel's monotonicity bound.
    """
    base = effective_midtone_gamma(paper, trim)
    return (base + snap_trims[0], base + snap_trims[1], base + snap_trims[2])


def _grade_trim_mult(grade: float, trim: float, c: Dict[str, Any]) -> float:
    """
    Per-layer ISO-R trim -> slope ratio: k ∝ 1/R, so a ΔR trim is the pure
    ratio R/(R+ΔR), with both grades clamped to the R ladder.
    """
    r0 = min(max(float(grade), float(c["iso_r_min"])), float(c["iso_r_max"]))
    r1 = min(max(r0 + float(trim), float(c["iso_r_min"])), float(c["iso_r_max"]))
    return r0 / r1



def grade_coupled_shape(slope_g: float, toe: float, shoulder: float) -> Tuple[float, float]:
    """
    Grade-coupled baseline toe/shoulder: hard grades (VC paper) physically have
    snappier toes and compressed shoulders. slope_norm = 0 at the softest grade,
    1 at the hardest. Single source of truth for the CPU engine, the GPU uniform
    packing and the curve chart — they must all draw the same knees.
    """
    c = EXPOSURE_CONSTANTS
    slope_norm = (float(slope_g) - float(c["slope_min"])) / (float(c["slope_max"]) - float(c["slope_min"]))
    slope_norm = min(max(slope_norm, 0.0), 1.0)
    toe_eff = float(toe) + float(c["toe_grade_strength"]) * slope_norm
    shoulder_eff = float(shoulder) + float(c["shoulder_grade_strength"]) * slope_norm
    return toe_eff, shoulder_eff


def effective_grade_range(
    auto_normalize_contrast: bool,
    floor_ceil_range: Optional[float],
    textural_range: Optional[float],
) -> Optional[float]:
    """
    Range fed to grade_to_slope. Auto Grade off: the measured floor-to-ceil range, a
    fixed paper. Auto Grade on: the paper gamma follows the negative's textural density
    scale, shrunk toward a normal negative's (Alkofer, US 4,731,671):
    effective = K * floor_ceil * ((1 - s) + s * nominal / textural),
    so the printed textural range is (1 - s) of the frame's own plus s of the norm,
    capped at max_overfill of the norm's print span.
    """
    c = EXPOSURE_CONSTANTS
    if not auto_normalize_contrast:
        return floor_ceil_range
    if textural_range is None or floor_ceil_range is None:
        return default_grade_range()
    measured = abs(float(textural_range))
    if measured < 1e-6:
        # Degenerate (near-flat) frame: let grade_to_slope's clamp cap the boost.
        return 3.5
    k = float(c["auto_grade_target"])
    q = float(c["auto_grade_nominal_range"]) / measured
    strength = float(c["auto_grade_strength"])
    # The print span of the textural range is capped at max_overfill of a normal
    # negative's: a grade whose paper range is far shorter than the negative's clips
    # both ends (the ISO R rule read as a floor on softness).
    factor = min((1.0 - strength) + strength * q, q * float(c["auto_grade_max_overfill"]))
    return k * abs(float(floor_ceil_range)) * factor


def _reference_linear_value(d_min: float = 0.0, paper: Optional[PaperProfile] = None, target: Optional[float] = None) -> float:
    """
    Straight-line density value v* that the base shoulder+toe bounds map onto `target`
    (default anchor_target_density). The reference tone is placed here so it
    prints at target, and the paper S-curve is centred here so the anchor is
    preserved. Closed form via inverse softplus at the base toe/shoulder sharpness.
    """
    c = effective_constants(paper)
    t = float(c["anchor_target_density"]) if target is None else float(target)
    d_max = float(c["d_max"])
    a_hl = float(c["shoulder_sharpness_base"])  # highlight (lower) bound
    a_sh = float(c["toe_sharpness_base"])  # shadow (upper) bound
    v1 = d_max - _inv_softplus_np(a_sh * (d_max - t)) / a_sh
    return float(d_min + _inv_softplus_np(a_hl * (v1 - d_min)) / a_hl)



def shadow_reach_slope(slope: float, anchor: float, shadow_point: float, d_min: float = 0.0, paper: Optional[PaperProfile] = None) -> float:
    """
    Auto Grade floor on the slope: the textured dark tail at `shadow_point` must print
    at least shadow_reach_density while the anchor stays at its target, so the slope
    is raised to the straight line through both when the grade alone falls short
    (Gindele, US 7,113,649; Ajewole, US 5,046,118). Never lowered.
    """
    c = effective_constants(paper)
    span = float(shadow_point) - float(anchor)
    if span <= 1e-6:
        return slope
    v_black = _reference_linear_value(d_min, paper, target=float(c["shadow_reach_density"]))
    needed = (v_black - _reference_linear_value(d_min, paper)) / span
    return float(min(max(slope, needed), float(c["slope_max"])))



def highlight_hold_offset(
    slope: float,
    pivot: float,
    highlight_point: float,
    d_min: float = 0.0,
    paper: Optional[PaperProfile] = None,
) -> float:
    """
    Auto Grade's soft exposure: the highlight zone burn (a highlight_density term) that
    lands the textured bright tail at `highlight_point` on highlight_hold_density when the
    straight line would print it brighter. Solved against the kernel's own zone weight at
    that tone, so the burn stays under the shoulder. Never lifts; 0 when the tone holds.
    """
    c = effective_constants(paper)
    target = float(c["highlight_hold_density"])
    if target <= 0.0:
        return 0.0
    v = float(slope) * (float(highlight_point) - float(pivot))
    v_hold = _reference_linear_value(d_min, paper, target=target)
    if v >= v_hold:
        return 0.0
    z_hi = float(c["anchor_target_density"]) + float(c["zone_density_highlight_offset"])
    w = 1.0 - _sigmoid_scalar(float(c["zone_density_sharpness"]) * (v - z_hi))
    return float(min((v_hold - v) / max(w, 1e-6), float(c["highlight_hold_max"])))



def compute_pivot(
    slope: float, density: float, d_min: float = 0.0, anchor: Optional[float] = None, paper: Optional[PaperProfile] = None
) -> float:
    """
    Fixed calibrated exposure: solve the curve pivot so the reference tone
    prints at anchor_target_density for the current effective slope — grade
    changes rotate around that reference tone instead of shifting brightness.
    The density slider offsets exposure around it. The reference tone defaults
    to assumed_anchor (a typical negative's normalized median); pass `anchor`
    to use a per-frame metered median (auto-exposure) instead.
    """
    c = effective_constants(paper)
    ref = c["assumed_anchor"] if anchor is None else anchor
    v_star = _reference_linear_value(d_min, paper)
    base = ref - v_star / slope
    return base + (1.0 - density) * c["density_multiplier"]


def normalize_refs(
    refs: Tuple[float, float, float],
    floors: Tuple[float, float, float],
    ceils: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Per-channel reference densities -> normalized [0, 1] position in the same
    floor->ceil stretch the image is normalized with. Shared by the CPU/GPU/chart
    call sites (Cast Removal shadow refs) so they can't drift.
    """
    epsilon = 1e-6
    out = []
    for ch in range(3):
        denom = ceils[ch] - floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        out.append((refs[ch] - floors[ch]) / denom)
    return (out[0], out[1], out[2])


def normalized_shadow_refs(bounds: Any, refs: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    """Shadow refs normalized against `bounds`, or None if either is missing."""
    if bounds is None or refs is None:
        return None
    return normalize_refs(refs, bounds.floors, bounds.ceils)


def normalized_neutral_axis(bounds: Any, refs: Any) -> Any:
    """(midtone, shadow, highlight) neutral refs normalized against `bounds`; highlight may be
    None (2-point), or the whole thing None if either core ref is missing."""
    if bounds is None or refs is None:
        return None
    mid, shadow, highlight = refs[0], refs[1], refs[2]  # refs may carry a trailing confidence
    norm = lambda r: normalize_refs(r, bounds.floors, bounds.ceils) if r is not None else None  # noqa: E731
    return (norm(mid), norm(shadow), norm(highlight))


def effective_cast_strength(strength: float, confidence: Optional[float]) -> float:
    """Applied cast-removal strength: the neutral-reference confidence biases the
    slider (clean greys → full, ambiguous → gentler); the slider trims on top."""
    if confidence is not None:
        return confidence * strength
    return strength


def cast_solve_inputs(
    bounds: Any,
    shadow_log_refs: Optional[Tuple[float, float, float]],
    neutral_axis_refs: Any,
    slider_strength: float,
) -> Tuple[float, Optional[Tuple[float, float, float]], Any]:
    """(effective strength, shadow_refs_norm, neutral_axis_norm) from raw metrics;
    single source of truth for the CPU processor and the chart."""
    shadow_refs_norm = normalized_shadow_refs(bounds, shadow_log_refs)
    neutral_axis_norm = normalized_neutral_axis(bounds, neutral_axis_refs)
    confidence = neutral_axis_refs[3] if neutral_axis_refs is not None else None
    return effective_cast_strength(slider_strength, confidence), shadow_refs_norm, neutral_axis_norm


def per_channel_curve_params(
    grade: float,
    density: float,
    auto_normalize_contrast: bool,
    strength: float,
    lum_range: Optional[float],
    shadow_refs_norm: Optional[Tuple[float, float, float]],
    textural_range: Optional[float],
    d_min: float = 0.0,
    anchor: Optional[float] = None,
    paper: Optional[PaperProfile] = None,
    neutral_axis_norm: Any = None,
    grade_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shadow_point: Optional[float] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-channel (slopes, pivots, curvatures); single source of truth for CPU/GPU/chart.
    Core: v = slope*(u - pivot) + curv*u² (curv 0 = a straight line).

    Cast Removal fits R/B to green's neutral axis (green is the reference; its pivot rides the
    luma anchor, so exposure is unchanged). neutral_axis_norm (midtone, shadow, highlight) ->
    quadratic through all three; (midtone, shadow) -> line; shadow_refs_norm only -> one-point
    shadow tie. Off / no refs (E6/B&W): one shared linear curve.
    """
    c = effective_constants(paper)
    # Per-channel slope multipliers for the paper dye-layer contrast crossover. The pivot is
    # re-solved per channel, so neutrals stay neutral and color diverges only away from the
    # midtone.
    cg = paper.channel_gamma if paper is not None else (1.0, 1.0, 1.0)
    if grade_trims != (0.0, 0.0, 0.0):
        # Per-layer ISO-R trims fold in as user channel_gammas, and the pivot re-solve keeps
        # the anchor neutral.
        cg = (
            cg[0] * _grade_trim_mult(grade, grade_trims[0], c),
            cg[1] * _grade_trim_mult(grade, grade_trims[1], c),
            cg[2] * _grade_trim_mult(grade, grade_trims[2], c),
        )
    slope_min = float(c["slope_min"])
    slope_max = float(c["slope_max"])
    r_eff = effective_grade_range(auto_normalize_contrast, lum_range, textural_range)
    base_slope = grade_to_slope(grade, r_eff)
    if auto_normalize_contrast and shadow_point is not None:
        ref = float(c["assumed_anchor"]) if anchor is None else float(anchor)
        base_slope = shadow_reach_slope(base_slope, ref, shadow_point, d_min=d_min, paper=paper)

    epsilon = 1e-6

    if strength > 0.0 and neutral_axis_norm is not None:
        # A line through the green-matched midtone and shadow, plus a highlight-driven
        # curvature when present, so highlights do not extrapolate past neutral. Clamped
        # monotonic.
        mid_norm, sh_norm, hl_norm = neutral_axis_norm
        limit = float(c["midtone_cast_max_offset"])
        curv_lim = float(c["neutral_axis_curv_max_ratio"])
        m_g, s_g = float(mid_norm[1]), float(sh_norm[1])
        slope_g = min(max(base_slope * cg[1], slope_min), slope_max)
        pivot_g = compute_pivot(slope_g, density, d_min=d_min, anchor=anchor, paper=paper)
        target = lambda g: slope_g * (g - pivot_g)  # noqa: E731  green's core at a green ref
        t_m, t_s = target(m_g), target(s_g)
        h_g = float(hl_norm[1]) if hl_norm is not None else None
        clamp_dev = lambda g, v: g + min(max(strength * (v - g), -limit), limit)  # noqa: E731

        slopes, pivots, curvs = [], [], []
        for ch in range(3):
            if ch == 1:
                slopes.append(slope_g)
                pivots.append(pivot_g)
                curvs.append(0.0)
                continue
            u_m = clamp_dev(m_g, float(mid_norm[ch]))
            u_s = clamp_dev(s_g, float(sh_norm[ch]))

            curv = 0.0
            if h_g is not None and hl_norm is not None:
                u_h = clamp_dev(h_g, float(hl_norm[ch]))
                # Leading coeff of the quadratic through the three green-matched points.
                m = np.array([[1.0, u_h, u_h * u_h], [1.0, u_m, u_m * u_m], [1.0, u_s, u_s * u_s]])
                try:
                    curv = float(np.linalg.solve(m, np.array([target(h_g), t_m, t_s]))[2])
                except np.linalg.LinAlgError:
                    curv = 0.0
                curv = min(max(curv, -curv_lim * slope_g), curv_lim * slope_g)

            # Re-pin midtone and shadow with this curvature. The mid is exact through the
            # pivot, like the two-point solve.
            du = u_m - u_s
            slope_ch = slope_g if abs(du) < epsilon else ((t_m - t_s) - curv * (u_m * u_m - u_s * u_s)) / du
            slope_ch = min(max(slope_ch * cg[ch], slope_min), slope_max)
            curv_ch = curv * cg[ch]
            pivot_ch = u_m - (t_m - curv_ch * u_m * u_m) / slope_ch if abs(slope_ch) > epsilon else pivot_g
            slopes.append(slope_ch)
            pivots.append(pivot_ch)
            curvs.append(curv_ch)
        return (slopes[0], slopes[1], slopes[2]), (pivots[0], pivots[1], pivots[2]), (curvs[0], curvs[1], curvs[2])

    if strength > 0.0 and shadow_refs_norm is not None:
        # Fallback one-point tie, used when there is no neutral axis: slope-tilt each channel
        # so its shadow ref lands on green's, with the luma anchor pinning the midtone.
        anchor_val = float(c["assumed_anchor"]) if anchor is None else float(anchor)
        limit = float(c["cast_removal_max_offset"])
        r_green = float(shadow_refs_norm[1])
        numer = anchor_val - r_green

        slopes = []
        pivots = []
        for ch in range(3):
            # Clamp the shadow cast before solving, bounding the correction.
            cast = min(max(strength * (r_green - float(shadow_refs_norm[ch])), -limit), limit)
            denom = anchor_val - (r_green - cast)
            if ch == 1 or abs(denom) < epsilon:
                slope_ch = base_slope
            else:
                slope_ch = base_slope * numer / denom
                slope_ch = min(max(slope_ch, slope_min), slope_max)
            slope_ch = min(max(slope_ch * cg[ch], slope_min), slope_max)
            slopes.append(slope_ch)
            pivots.append(compute_pivot(slope_ch, density, d_min=d_min, anchor=anchor, paper=paper))
        return (slopes[0], slopes[1], slopes[2]), (pivots[0], pivots[1], pivots[2]), (0.0, 0.0, 0.0)

    # Base curve: Cast Removal off, or on with no measured refs (E6, B&W, no neutrals).
    s0 = min(max(base_slope * cg[0], slope_min), slope_max)
    s1 = min(max(base_slope * cg[1], slope_min), slope_max)
    s2 = min(max(base_slope * cg[2], slope_min), slope_max)
    p0 = compute_pivot(s0, density, d_min=d_min, anchor=anchor, paper=paper)
    p1 = compute_pivot(s1, density, d_min=d_min, anchor=anchor, paper=paper)
    p2 = compute_pivot(s2, density, d_min=d_min, anchor=anchor, paper=paper)
    return (s0, s1, s2), (p0, p1, p2), (0.0, 0.0, 0.0)


def neutral_axis_affine(
    neutral_axis_norm: Any,
    strength: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-channel (gain, offset) on normalized density that lands each channel's neutral
    refs on green's — Cast Removal for the transparency transfer, which has no per-channel
    slope to re-solve. Green is identity, and so is a missing axis or zero strength.

    The clamped-deviation convention matches per_channel_curve_params, so `strength` means
    the same thing on both curves: green's ref moves a fraction of the way toward the
    channel's own, and that moved point is what maps onto green.
    """
    unit = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    if strength <= 0.0 or neutral_axis_norm is None:
        return unit
    mid_norm, sh_norm = neutral_axis_norm[0], neutral_axis_norm[1]
    c = EXPOSURE_CONSTANTS
    limit = float(c["midtone_cast_max_offset"])
    gain_max = float(c["cast_affine_gain_limit"])
    epsilon = 1e-6

    m_g, s_g = float(mid_norm[1]), float(sh_norm[1])
    clamp_dev = lambda g, v: g + min(max(strength * (v - g), -limit), limit)  # noqa: E731

    gains, offsets = [], []
    for ch in range(3):
        if ch == 1:
            gains.append(1.0)
            offsets.append(0.0)
            continue
        u_m = clamp_dev(m_g, float(mid_norm[ch]))
        u_s = clamp_dev(s_g, float(sh_norm[ch]))
        du = u_m - u_s
        if abs(du) < epsilon:
            gains.append(1.0)
            offsets.append(0.0)
            continue
        gain = min(max((m_g - s_g) / du, 1.0 / gain_max), gain_max)
        # Offset re-solved after the clamp, so the midtone stays exact.
        gains.append(gain)
        offsets.append(m_g - gain * u_m)
    return (gains[0], gains[1], gains[2]), (offsets[0], offsets[1], offsets[2])



def filtration_offsets(wb_cmy: Tuple[float, float, float], bounds: Any) -> Tuple[float, float, float]:
    """
    WB sliders as normalized-space offsets: slider · cmy_max_density is an
    absolute density (1.0 = 20cc), divided by each channel's stretch range so
    the same slider prints the same filtration on every frame. abs() keeps the
    slider direction uniform across C-41/E-6. Range 1 when bounds are None.
    """
    cmy_max = float(EXPOSURE_CONSTANTS["cmy_max_density"])
    out = []
    for ch in range(3):
        d = float(wb_cmy[ch]) * cmy_max
        if bounds is not None:
            d = d / max(abs(bounds.ceils[ch] - bounds.floors[ch]), 1e-6)
        out.append(d)
    return (out[0], out[1], out[2])



def cmy_to_density(val: float, log_range: float = 1.0) -> float:
    """
    Converts a CMY slider value (-1.0..1.0) to a physical density shift (D).
    """
    absolute_density = val * EXPOSURE_CONSTANTS["cmy_max_density"]
    return float(absolute_density / max(log_range, 1e-6))


def density_to_cmy(density: float, log_range: float = 1.0) -> float:
    """
    Converts a physical density shift (D) back to a normalized CMY slider value.
    """
    absolute_density = density * log_range
    return float(absolute_density / EXPOSURE_CONSTANTS["cmy_max_density"])


TEMP_REF_KELVIN = 5500.0
TEMP_MIN_KELVIN, TEMP_MAX_KELVIN = 3000.0, 12000.0
# Slider units per mired, red-anchored so wb_cyan stays 0: Wien slopes
# c2*log10(e)*(1/lambda - 1/lambda_R) at roughly 460/550/690 nm, over a nominal effective
# print gamma and cmy_max_density. The Kelvin readout is therefore nominal, not
# colorimetric. Calibration knobs: retune by eye.
_TEMP_K_MAGENTA = 0.0029
_TEMP_K_YELLOW = 0.0057


def wb_to_kelvin(magenta: float, yellow: float) -> float:
    """
    Measured print temperature: least-squares projection of the global (M, Y)
    pair onto the Planckian (mired) direction; 5500K at neutral.
    """
    km, ky = _TEMP_K_MAGENTA, _TEMP_K_YELLOW
    dmu = (km * magenta + ky * yellow) / (km * km + ky * ky)
    # Clamp in the mired domain: a Kelvin-domain clamp breaks when mu goes negative.
    mu = min(max(1e6 / TEMP_REF_KELVIN + dmu, 1e6 / TEMP_MAX_KELVIN), 1e6 / TEMP_MIN_KELVIN)
    return float(1e6 / mu)


def kelvin_to_wb(kelvin: float, magenta: float, yellow: float) -> Tuple[float, float]:
    """
    Moves (M, Y) along the Planckian direction to land on `kelvin`, keeping the
    off-locus green-magenta tint component untouched. Clips to slider range.
    """
    km, ky = _TEMP_K_MAGENTA, _TEMP_K_YELLOW
    kelvin = min(max(kelvin, TEMP_MIN_KELVIN), TEMP_MAX_KELVIN)
    dmu_cur = (km * magenta + ky * yellow) / (km * km + ky * ky)
    d = (1e6 / kelvin - 1e6 / TEMP_REF_KELVIN) - dmu_cur
    m2 = min(max(magenta + km * d, -1.0), 1.0)
    y2 = min(max(yellow + ky * d, -1.0), 1.0)
    return float(m2), float(y2)


def calculate_wb_shifts(sampled_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts to neutralize sampled color in positive space.
    """
    r, g, b = np.clip(sampled_rgb, 1e-6, 1.0)
    d_m = np.log10(g) - np.log10(r)
    d_y = np.log10(b) - np.log10(r)

    shift_m = density_to_cmy(d_m)
    shift_y = density_to_cmy(d_y)

    return float(shift_m), float(shift_y)


def calculate_wb_shifts_from_log(sampled_log_rgb: np.ndarray, bounds: Any = None) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts from data in Negative Log-Density space.
    `bounds` converts the deviation to absolute density (see filtration_offsets).
    """
    r, g, b = sampled_log_rgb[:3]
    d_m = r - g
    d_y = r - b

    rng = lambda ch: abs(bounds.ceils[ch] - bounds.floors[ch]) if bounds is not None else 1.0  # noqa: E731
    shift_m = density_to_cmy(d_m, rng(1))
    shift_y = density_to_cmy(d_y, rng(2))

    return float(shift_m), float(shift_y)


# ── Vectorized print-curve kernel ────────────────────────────────────────────

CHUNK_PIXELS = 1 << 20
_LN10 = float(np.log(10.0))


def _expit_f32(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _curve_chunk(
    x: np.ndarray,
    ch_params: Any,
    zone: Tuple[float, float, float, float],
    shadow_density: float,
    highlight_density: float,
    v_star: float,
    gamma_width: float,
) -> np.ndarray:
    """One channel's density for a flat float32 chunk (curve core, S-curve, zones, toe/shoulder)."""
    (pivot, slope, curv, off, mg, a_hl, a_sh, d_min_eff, d_max_eff) = ch_params
    v = quadratic_core(slope, pivot, curv, x + np.float32(off))
    if mg != 0.0:
        v = v + np.float32(mg * gamma_width) * np.tanh((v - np.float32(v_star)) / np.float32(gamma_width))
    zk, zsh, zhi = zone[1], zone[2], zone[3]
    if shadow_density != 0.0 or highlight_density != 0.0:
        w_sh = _expit_f32(np.float32(zk) * (v - np.float32(zsh)))
        w_hi = 1.0 - _expit_f32(np.float32(zk) * (v - np.float32(zhi)))
        v = v + np.float32(shadow_density) * w_sh + np.float32(highlight_density) * w_hi
    v1 = d_min_eff + np.logaddexp(np.float32(0.0), a_hl * (v - d_min_eff)) / a_hl
    return d_max_eff - np.logaddexp(np.float32(0.0), a_sh * (d_max_eff - v1)) / a_sh


def apply_characteristic_curve(
    img: ImageBuffer,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    toe_width: float = 2.5,
    shoulder: float = 0.0,
    shoulder_width: float = 2.5,
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    d_min: float = 0.0,
    midtone_gamma: Optional[float] = None,
    curvatures: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    paper: Optional[PaperProfile] = None,
    bpc: bool = False,
    snap_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shadow_density: float = 0.0,
    highlight_density: float = 0.0,
    dye_separation: float = 1.0,
    out: Optional[np.ndarray] = None,
    chunk_pixels: int = CHUNK_PIXELS,
) -> ImageBuffer:
    """Print curve on a normalized (H,W,3) float32 image -> linear reflectance in [0,1].

    params_* are (pivot, slope) per channel (see per_channel_curve_params, which returns
    (slopes, pivots, curvs): note the order here is pivot first, like NegPy). `toe`/`shoulder`
    are the grade-coupled values (grade_coupled_shape). A 2-D (H,W) or (H,W,1) input runs the
    green-channel curve only (B&W), with no dye mixing. `out` may be a preallocated float32
    (or uint16-compatible float) buffer of the input's shape; rows are processed in chunks
    of ~chunk_pixels so no full-size temporaries are held.
    """
    c = effective_constants(paper)
    ts = float(c["toe_shoulder_strength"])
    if midtone_gamma is None:
        midtone_gamma = float(c["paper_midtone_gamma"])
    v_star = _reference_linear_value(d_min, paper)
    gamma_width = float(c["paper_gamma_width"])
    zone = (
        float(c["anchor_target_density"]),
        float(c["zone_density_sharpness"]),
        float(c["anchor_target_density"]) + float(c["zone_density_shadow_offset"]),
        float(c["anchor_target_density"]) + float(c["zone_density_highlight_offset"]),
    )
    eps = 1e-6
    d_max = float(c["d_max"])
    wr = float(c["toeshoulder_width_ref"])
    toe_h, sh_h = float(c["toe_height"]), float(c["shoulder_height"])
    dmin_rgb = paper_dmin_rgb(d_min, paper)
    toe3, sh3 = per_channel_toe_shoulder(toe, shoulder)
    tw3, sw3 = per_channel_widths(toe_width, shoulder_width)
    plist = (params_r, params_g, params_b)

    ch_params, bpc_black = [], []
    for ch in range(3):
        t = toe3[ch] * ts
        a_hl = float(c["shoulder_sharpness_base"]) * wr / max(sw3[ch], eps)
        a_sh = float(c["toe_sharpness_base"]) * wr / max(tw3[ch], eps)
        if t >= 0.0:
            d_max_base = d_max - t * toe_h
        else:
            d_max_base = d_max
            a_sh = a_sh * (1.0 - t * 4.0)
        dmn = max(dmin_rgb[ch] + sh3[ch] * ts * sh_h, 0.0)
        dmx = max(d_max_base, dmn + 0.1)
        db = d_max + t * toe_h if t < 0.0 else d_max
        bpc_black.append(10.0**-db)
        ch_params.append(
            (
                float(plist[ch][0]),
                float(plist[ch][1]),
                float(curvatures[ch]),
                float(cmy_offsets[ch]),
                float(midtone_gamma) + float(snap_trims[ch]),
                a_hl,
                a_sh,
                dmn,
                dmx,
            )
        )

    gray = img.ndim == 2 or img.shape[-1] == 1
    src = np.asarray(img, dtype=np.float32)
    if gray:
        src = src.reshape(src.shape[0], src.shape[1], 1)
    h, w, nch = src.shape
    if out is None:
        out = np.empty(src.shape if not gray else img.shape, dtype=np.float32)
    out_view = out.reshape(h, w, nch)

    dye = resolve_dye_matrix(paper)
    sat = resolve_saturation_matrix(per_channel_dye_separation(dye_separation))
    composed = None if gray else compose_density_matrices(dye, sat)
    mix = None if composed is None else composed.astype(np.float32)
    dmin32 = np.array(dmin_rgb, dtype=np.float32)

    rows = max(1, chunk_pixels // max(w, 1))
    for y0 in range(0, h, rows):
        blk = src[y0 : y0 + rows]
        n = blk.shape[0] * w
        flat = blk.reshape(n, nch)
        dens = np.empty((n, 3 if not gray else 1), dtype=np.float32)
        chans = range(3) if not gray else (1,)
        for i, ch in enumerate(chans):
            dens[:, i] = _curve_chunk(
                flat[:, i], ch_params[ch], zone, shadow_density, highlight_density, v_star, gamma_width
            )
        if mix is not None:
            e = dens - dmin32
            dens = dmin32 + e @ mix.T
        tr = np.exp(dens * np.float32(-_LN10))
        if bpc:
            bk = np.array([bpc_black[ch] for ch in (range(3) if not gray else (1,))], dtype=np.float32)
            tr = (tr - bk) / (1.0 - bk)
        np.clip(tr, 0.0, 1.0, out=tr)
        out_view[y0 : y0 + blk.shape[0]] = tr.reshape(blk.shape[0], w, nch)
    return out
