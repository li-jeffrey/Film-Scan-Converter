"""Slide (transparency) transfer path and camera colour matrix.

Ported from NegPy (GPL-3.0): features/transparency/logic.py and
features/process/capture_color.py, vectorized NumPy, no numba. Config objects are duck-typed
(attribute names follow NegPy's ExposureConfig). Functions that need the print-curve helpers
(grade_to_slope, effective_grade_range) import them lazily from .print_curve.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import (
    EXPOSURE_CONSTANTS,
    SEPARATION_CAP_LEVEL,
    SEPARATION_CAP_SOFTNESS,
    SEPARATION_CAP_SPREAD,
    TRANSFER_CONSTANTS,
    TRANSFER_DENSITY_RANGE,
    XYZ_TO_ADOBE_RGB,
)

_WORKING_TO_XYZ = np.linalg.inv(XYZ_TO_ADOBE_RGB)

# ---------------------------------------------------------------- capture colour


def camera_to_working_matrix(
    cam_xyz: Optional[Sequence[Sequence[float]]],
    camera_wb: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    """Working-space(Adobe RGB, linear)-from-camera 3x3 from libraw's rgb_xyz_matrix.

    Row-normalizes the working->cam matrix and then inverts (dcraw order). `camera_wb` folds
    the as-shot multipliers back in for buffers decoded without white balance. None when no
    usable matrix exists.
    """
    if cam_xyz is None:
        return None
    m = np.asarray(cam_xyz, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 3 or m.shape[0] < 3:
        return None
    m = m[:3, :]
    if not np.all(np.isfinite(m)) or abs(float(np.linalg.det(m))) < 1e-9:
        return None
    forward = m @ _WORKING_TO_XYZ
    sums = forward.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(sums)) or np.any(np.abs(sums) < 1e-9):
        return None
    out = np.linalg.pinv(forward / sums)
    if not np.all(np.isfinite(out)):
        return None
    if camera_wb is not None:
        wb = np.asarray(camera_wb, dtype=np.float64).reshape(-1)[:3]
        if wb.shape[0] == 3 and np.all(np.isfinite(wb)) and float(wb.min()) > 0.0 and float(wb[1]) > 0.0:
            out = out @ np.diag(wb / wb[1])
    return np.ascontiguousarray(out, dtype=np.float32)


def wb_only_cam_xyz(cam_xyz: Optional[Sequence[Sequence[float]]]) -> Optional[list]:
    """Stand-in cam_xyz for a profiled source: yields an identity primaries rotation."""
    if cam_xyz is None:
        return None
    return XYZ_TO_ADOBE_RGB.tolist()


def apply_camera_matrix(img: np.ndarray, matrix: Optional[np.ndarray], chunk: int = 1 << 20) -> np.ndarray:
    """Camera primaries -> working space. None passes through. Negatives are kept."""
    if matrix is None:
        return img
    flat = img.reshape(-1, 3)
    m_t = np.asarray(matrix, dtype=np.float32).T
    out = np.empty(flat.shape, dtype=np.float32)
    for i in range(0, flat.shape[0], chunk):
        out[i : i + chunk] = flat[i : i + chunk].astype(np.float32, copy=False) @ m_t
    return out.reshape(img.shape)


_LIGHTBOX_PCT = 99.5
_LIGHTBOX_TARGET = 0.95


def lightbox_level(img: np.ndarray, matrix: Optional[np.ndarray]) -> Optional[float]:
    """Scalar gain putting the frame's P99.5 tone at 0.95; None when there is no signal."""
    flat = img.reshape(-1, 3)
    step = max(1, flat.shape[0] // 200_000)
    sample = flat[::step].astype(np.float64)
    if matrix is not None:
        sample = sample @ np.asarray(matrix, dtype=np.float64).T
    white = float(np.percentile(sample, _LIGHTBOX_PCT))
    if not np.isfinite(white) or white <= 1e-6:
        return None
    return _LIGHTBOX_TARGET / white


# ---------------------------------------------------------------- transfer curve


def _per_channel_toe_shoulder(toe, shoulder, toe_trims=(0.0,) * 3, shoulder_trims=(0.0,) * 3):
    def c(v):
        return min(max(v, -1.0), 1.0)

    return (
        tuple(c(toe + t) for t in toe_trims),
        tuple(c(shoulder + t) for t in shoulder_trims),
    )


def _per_channel_widths(toe_w, shoulder_w, toe_trims=(0.0,) * 3, shoulder_trims=(0.0,) * 3):
    def c(v):
        return min(max(v, 0.1), 5.0)

    return (
        tuple(c(toe_w + t) for t in toe_trims),
        tuple(c(shoulder_w + t) for t in shoulder_trims),
    )


def _per_channel_dye_separation(sep, trims=(0.0,) * 3):
    return tuple(min(max(sep + t, 0.0), 3.0) for t in trims)


def separation_damping_gain_np(k: float, damping: float, chroma, ref_spread: float):
    """Per-pixel dye-separation gain tapered by chroma."""
    if k <= 0.0:
        return np.zeros_like(chroma, dtype=np.float32)
    h = (np.float32(ref_spread) - chroma) / (np.float32(ref_spread) + chroma)
    kf = np.power(np.float32(k), (1.0 - np.float32(damping)) + np.float32(damping) * h, dtype=np.float32)
    return np.minimum(kf, np.float32(3.0))


def _fast_sigmoid(x: float) -> float:
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    z = float(np.exp(x))
    return z / (1.0 + z)


def zone_geometry() -> Tuple[float, float, float]:
    """(shadow centre, highlight centre, sharpness) for Zone Density, in transfer density."""
    c = EXPOSURE_CONSTANTS
    d_min, d_max = float(c["d_min"]), float(c["d_max"])
    span = d_max - d_min
    anchor = float(c["anchor_target_density"])
    shadow = (anchor + float(c["zone_density_shadow_offset"]) - d_min) / span
    highlight = (anchor + float(c["zone_density_highlight_offset"]) - d_min) / span
    return (
        shadow * TRANSFER_DENSITY_RANGE,
        highlight * TRANSFER_DENSITY_RANGE,
        float(c["zone_density_sharpness"]) * span / TRANSFER_DENSITY_RANGE,
    )


def wb_split_geometry() -> Tuple[float, float]:
    """(centre, sharpness) for the Shadows/Highlights white-balance split."""
    c = EXPOSURE_CONSTANTS
    d_min, d_max = float(c["d_min"]), float(c["d_max"])
    span = d_max - d_min
    centre = (float(c["anchor_target_density"]) - d_min) / span * TRANSFER_DENSITY_RANGE
    return centre, 3.0 * span / TRANSFER_DENSITY_RANGE


def display_rendering(scene_linear: np.ndarray) -> np.ndarray:
    """Scene-linear -> display-linear (Narkowicz ACES RRT+sRGB ODT fit)."""
    x = np.maximum(np.asarray(scene_linear, dtype=np.float32), 0.0)
    num = x * (np.float32(2.51) * x + np.float32(0.03))
    den = x * (np.float32(2.43) * x + np.float32(0.59)) + np.float32(0.14)
    return np.clip(num / np.maximum(den, np.float32(1e-8)), 0.0, 1.0).astype(np.float32)


def _sigmoid(x):
    return (0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float32)))).astype(np.float32)


ZONE_BLACK_TAPER = 1.0


def _black_taper(d, density_range: float):
    t = np.clip((np.float32(density_range) - d) / np.float32(ZONE_BLACK_TAPER), 0.0, 1.0)
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)


def _softplus(x, width: float):
    t = x / width
    return (width * (np.logaddexp(0.0, -np.abs(t)) + np.maximum(t, 0.0))).astype(np.float32)


def transfer_widths(config):
    """Per-channel knee softness (density) from the Toe/Shoulder Width sliders."""
    c = TRANSFER_CONSTANTS
    scale = float(c["transfer_knee_width"]) / float(c["transfer_width_ref"])
    tw3, sw3 = _per_channel_widths(float(config.toe_width), float(config.shoulder_width))
    return tuple(v * scale for v in tw3), tuple(v * scale for v in sw3)


def transfer_assumed_anchor() -> float:
    return float(TRANSFER_CONSTANTS["transfer_contrast_pivot"]) / TRANSFER_DENSITY_RANGE


def transfer_shadow_reach_density() -> float:
    c = EXPOSURE_CONSTANTS
    d_min, span = float(c["d_min"]), float(c["d_max"]) - float(c["d_min"])
    return (float(c["shadow_reach_density"]) - d_min) / span * TRANSFER_DENSITY_RANGE


def transfer_highlight_hold_density() -> float:
    c = EXPOSURE_CONSTANTS
    d_min, span = float(c["d_min"]), float(c["d_max"]) - float(c["d_min"])
    return (float(c["highlight_hold_density"]) - d_min) / span * TRANSFER_DENSITY_RANGE


def transfer_auto_terms(
    exposure,
    manual_offset: float,
    manual_contrast: float,
    textural_range: Optional[float],
    anchor: Optional[float],
    shadow_point: Optional[float],
    highlight_point: Optional[float],
) -> Tuple[float, float, float]:
    """(exposure_offset, contrast, highlight_density_auto) with Auto Density/Grade folded in.

    `exposure` needs .auto_exposure, .auto_normalize_contrast, .grade.
    """
    from .print_curve import effective_grade_range, grade_to_slope

    c = TRANSFER_CONSTANTS
    pivot = float(c["transfer_contrast_pivot"])
    R = TRANSFER_DENSITY_RANGE

    auto_offset = 0.0
    if exposure.auto_exposure and anchor is not None:
        auto_offset = float(anchor) * R - pivot
    offset = manual_offset + auto_offset

    contrast = manual_contrast
    if exposure.auto_normalize_contrast and textural_range is not None:
        effective_range = effective_grade_range(True, R, textural_range)
        k_ref = grade_to_slope(float(c["transfer_grade_ref"]), R)
        if k_ref > 1e-9:
            contrast = grade_to_slope(float(exposure.grade), effective_range) / k_ref

    if exposure.auto_normalize_contrast and anchor is not None and shadow_point is not None:
        span = float(shadow_point) - float(anchor)
        if span > 1e-6:
            d_shadow = float(shadow_point) * R - offset
            denom = d_shadow - pivot
            if denom > 1e-6:
                needed = (transfer_shadow_reach_density() - pivot) / denom
                contrast = min(max(contrast, needed), float(EXPOSURE_CONSTANTS["slope_max"]))

    highlight_auto = 0.0
    if exposure.auto_normalize_contrast and highlight_point is not None:
        target = transfer_highlight_hold_density()
        d_highlight = pivot + (float(highlight_point) * R - offset - pivot) * contrast
        if d_highlight < target:
            _sh_c, hi_c, k_zone = zone_geometry()
            w_hi = 1.0 - _fast_sigmoid(k_zone * (d_highlight - hi_c))
            highlight_auto = min((target - d_highlight) / max(w_hi, 1e-6), float(EXPOSURE_CONSTANTS["highlight_hold_max"]))
    return offset, contrast, highlight_auto


def transfer_curve_params(config):
    """(exposure_density_offset, contrast, toe3, shoulder3). `config` needs density, grade, toe, shoulder."""
    c = TRANSFER_CONSTANTS
    stops = (1.0 - float(config.density)) * float(c["transfer_density_stops"])
    exposure_offset = stops * float(np.log10(2.0))
    grade = float(config.grade)
    contrast = float(c["transfer_grade_ref"]) / grade if grade > 1e-6 else 1.0
    toe3, sh3 = _per_channel_toe_shoulder(float(config.toe), float(config.shoulder))
    return exposure_offset, contrast, toe3, sh3


def apply_transfer_curve(
    img_norm: np.ndarray,
    exposure_offset: float,
    contrast: float,
    toe: Tuple[float, float, float],
    shoulder: Tuple[float, float, float],
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    toe_widths: Optional[Tuple[float, float, float]] = None,
    shoulder_widths: Optional[Tuple[float, float, float]] = None,
    density_range: float = TRANSFER_DENSITY_RANGE,
    shadow_density: float = 0.0,
    highlight_density: float = 0.0,
    shadow_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cast_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cast_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    positive_source: bool = False,
    separation: float = 1.0,
    separation_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    damping: float = 0.0,
) -> np.ndarray:
    """Normalized log density -> display-linear positive (see NegPy for derivation).

    With every control at its neutral value the scene stage is an exact identity
    (10**-(range*n)); `display_rendering` then applies. `positive_source` skips baseline
    gain and display rendering.
    """
    c = TRANSFER_CONSTANTS
    base_width = float(c["transfer_knee_width"])
    tw3 = toe_widths or (base_width,) * 3
    sw3 = shoulder_widths or (base_width,) * 3
    pivot = float(c["transfer_contrast_pivot"])
    toe_knee = float(c["transfer_toe_knee"])
    sh_knee = float(c["transfer_shoulder_knee"])

    def shape(d: np.ndarray, ch: int) -> np.ndarray:
        if cast_gain[ch] != 1.0 or cast_offset[ch] != 0.0:
            d = d * np.float32(cast_gain[ch]) + np.float32(cast_offset[ch])
        if exposure_offset != 0.0 or cmy_offsets[ch] != 0.0:
            d = d - np.float32(exposure_offset) + np.float32(cmy_offsets[ch] * density_range)
        if contrast != 1.0:
            d = np.float32(pivot) + (d - np.float32(pivot)) * np.float32(contrast)
        if shadow_cmy[ch] != 0.0 or highlight_cmy[ch] != 0.0:
            wb_c, wb_k = wb_split_geometry()
            w_sh = _sigmoid(np.float32(wb_k) * (d - np.float32(wb_c)))
            w_hi = np.float32(1.0) - w_sh
            d = d + np.float32(shadow_cmy[ch]) * w_sh + np.float32(highlight_cmy[ch]) * w_hi
        if shadow_density != 0.0 or highlight_density != 0.0:
            sh_c, hi_c, k = zone_geometry()
            w_sh = _sigmoid(np.float32(k) * (d - np.float32(sh_c)))
            w_sh = w_sh * _black_taper(d, density_range)
            w_hi = np.float32(1.0) - _sigmoid(np.float32(k) * (d - np.float32(hi_c)))
            d = d + np.float32(shadow_density) * w_sh + np.float32(highlight_density) * w_hi
        if toe[ch] != 0.0:
            d = d - np.float32(toe[ch]) * _softplus(d - np.float32(toe_knee), tw3[ch])
        if shoulder[ch] != 0.0:
            d = d + np.float32(shoulder[ch]) * _softplus(np.float32(sh_knee) - d, sw3[ch])
        return d

    n = np.asarray(img_norm, dtype=np.float32)
    capture = n * np.float32(density_range)
    dens = np.empty_like(n)
    for ch in range(3):
        dens[..., ch] = shape(capture[..., ch], ch)

    sep_k3 = _per_channel_dye_separation(separation, separation_trims)
    if sep_k3 != (1.0, 1.0, 1.0):
        d_lo = capture.min(axis=-1, keepdims=True)
        cap = np.float32(SEPARATION_CAP_LEVEL) + _softplus(
            d_lo + np.float32(SEPARATION_CAP_SPREAD - SEPARATION_CAP_LEVEL), SEPARATION_CAP_SOFTNESS
        )
        capped = cap - _softplus(cap - capture, SEPARATION_CAP_SOFTNESS)
        capped = np.stack([shape(capped[..., ch], ch) for ch in range(3)], axis=-1)
        d_ref = capped.mean(axis=-1, keepdims=True)
        e = dens - d_ref
        if damping > 0.0:
            ce = capped - d_ref
            chroma = np.sqrt(
                ((ce[..., 0] - ce[..., 1]) ** 2 + (ce[..., 1] - ce[..., 2]) ** 2 + (ce[..., 0] - ce[..., 2]) ** 2) / 3.0
            )
            ref_spread = float(EXPOSURE_CONSTANTS["separation_damping_ref_spread"])
            k_eff = np.stack([separation_damping_gain_np(sep_k3[ch], damping, chroma, ref_spread) for ch in range(3)], axis=-1)
            dens = d_ref + k_eff * e
        else:
            dens = d_ref + np.asarray(sep_k3, dtype=np.float32) * e

    out = np.power(np.float32(10.0), -dens, dtype=np.float32)
    if positive_source:
        return np.clip(out, 0.0, 1.0)
    gain = np.float32(2.0 ** float(c["transfer_baseline_ev"]))
    return display_rendering(out * gain)


def transfer_point_offsets(white_offsets, black_offsets):
    """White/Black Point on the fixed window, negated (takes per-channel triples)."""
    return tuple(-v for v in white_offsets), tuple(-v for v in black_offsets)


def transfer_bounds(density_range: float = TRANSFER_DENSITY_RANGE):
    """Fixed (floors, ceils): decoder white level down `density_range` decades."""
    return (0.0, 0.0, 0.0), (-density_range, -density_range, -density_range)
