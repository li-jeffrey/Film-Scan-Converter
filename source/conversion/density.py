"""Density-domain metering: log10, block-median grid, bounds and meters.

Ported from NegPy ``features/exposure/normalization.py`` (GPL-3.0, see NOTICE.md).
Vectorized NumPy replacement for the numba kernels; roll-level pooling, the
contrast mask and the analysis-region helpers are intentionally not ported.
"""

from typing import Optional, Tuple

import numpy as np

from .constants import EXPOSURE_CONSTANTS

ImageBuffer = np.ndarray

LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722


def _normalize_log_image(img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray) -> np.ndarray:
    """Log -> ~0..1 linear stretch, unclamped. Handles f < c and f > c."""
    epsilon = 1e-6
    delta = (ceils - floors).astype(np.float32)
    denom = np.where(np.abs(delta) < epsilon, np.where(delta >= 0, epsilon, -epsilon), delta).astype(np.float32)
    return ((img_log - floors.astype(np.float32)) / denom).astype(np.float32, copy=False)


class LogNegativeBounds:
    """D-min / D-max container."""

    def __init__(self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]):
        self.floors = floors
        self.ceils = ceils


def get_analysis_crop(img: ImageBuffer, buffer_ratio: float) -> ImageBuffer:
    """Center crop excluding `buffer_ratio` (0..0.3) of each border."""
    if buffer_ratio <= 0:
        return img
    h, w = img.shape[:2]
    safe_buffer = min(max(buffer_ratio, 0.0), 0.3)
    cut_h = int(h * safe_buffer)
    cut_w = int(w * safe_buffer)
    return img[cut_h : h - cut_h, cut_w : w - cut_w]


def _block_median_grid(img_log: ImageBuffer) -> ImageBuffer:
    """Block-median prefilter to a fixed target grid (speculars/dust vanish in the median)."""
    h, w = img_log.shape[:2]
    grid = int(EXPOSURE_CONSTANTS["analysis_grid"])
    b = int(np.ceil(max(h, w) / grid))
    if b <= 1 or h < b or w < b:
        return img_log

    hb, wb = (h // b) * b, (w // b) * b
    arr = img_log[:hb, :wb]
    c = arr.shape[2]
    v = arr.reshape(hb // b, b, wb // b, b, c)
    if b == 2:
        p00, p01, p10, p11 = v[:, 0, :, 0], v[:, 0, :, 1], v[:, 1, :, 0], v[:, 1, :, 1]
        s = p00.astype(np.float64) + p01 + p10 + p11
        mn = np.minimum(np.minimum(p00, p01), np.minimum(p10, p11))
        mx = np.maximum(np.maximum(p00, p01), np.maximum(p10, p11))
        return ((s - mn - mx) * 0.5).astype(arr.dtype, copy=False)
    return np.median(v, axis=(1, 3))


def prefilter_log_grid(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> ImageBuffer:
    """Shared meter prefilter (log10 -> crop -> block-median grid)."""
    img_log = to_log_density(image)
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    return _block_median_grid(img_log)


def measure_clip_fractions(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> tuple[float, float, float]:
    """Per-channel fraction of pixels at/above the sensor-white clip level (linear input)."""
    img = image
    if roi:
        y1, y2, x1, x2 = roi
        img = img[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img = get_analysis_crop(img, analysis_buffer)
    img = img[::4, ::4]
    level = float(EXPOSURE_CONSTANTS["scan_clip_level"])
    clipped = np.mean(img >= level, axis=(0, 1))
    return (float(clipped[0]), float(clipped[1]), float(clipped[2]))


def unmix_log_image(img_log: ImageBuffer, matrix: Optional[np.ndarray]) -> ImageBuffer:
    """Apply the unmix matrix to a (H, W, 3) log-density image; identity when None."""
    if matrix is None:
        return img_log
    m = matrix.astype(np.float32)
    flat = img_log.astype(np.float32, copy=False).reshape(-1, 3)
    out = np.empty_like(flat)
    chunk = 1 << 20
    for i in range(0, flat.shape[0], chunk):
        out[i : i + chunk] = flat[i : i + chunk] @ m.T
    return out.reshape(img_log.shape)


def sorted_channel_grid(img_log: ImageBuffer) -> np.ndarray:
    """
    Per-channel sorted values of a prefiltered grid, shape (n, 3). The clip percentiles
    index into it, so a clip drag re-reads one sort instead of re-partitioning the grid.
    """
    return np.sort(img_log.reshape(-1, 3), axis=0)


def percentile_from_sorted(sorted_grid: np.ndarray, q: float) -> np.ndarray:
    """
    The three channels' `np.percentile(grid[:, :, ch], q)`, read off `sorted_channel_grid`.

    Bit-exact needs numpy's arithmetic, not its definition: virtual index `(n-1)*(q/100)`
    in float64, the far half interpolated back from the upper sample, and a weight that
    keeps `q`'s own type -- numpy interpolates in float32 for a Python `q` and float64 for
    an `np.float64` one, and both kinds of caller exist here.
    """
    n = sorted_grid.shape[0]
    virtual = (n - 1) * np.true_divide(q, 100)
    lo = min(max(int(np.floor(virtual)), 0), n - 1)
    hi = min(lo + 1, n - 1)
    t = virtual - lo
    if type(q) in (int, float):
        t = float(t)
    a, b = sorted_grid[lo], sorted_grid[hi]
    diff = b - a
    return b - diff * (1 - t) if t >= 0.5 else a + diff * t


def to_log_density(image: ImageBuffer) -> ImageBuffer:
    """
    Linear scan -> log10 density, the domain every meter reads.

    fmin/fmax rather than clip: they drop a NaN in favour of the bound, so one clamp covers
    the NaN and infinity fixup as well.
    """
    epsilon = 1e-6
    return np.log10(np.fmin(np.fmax(image, epsilon), 1.0))


def measure_shadow_refs_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    sorted_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Per-channel shadow reference density: a high percentile of the prefiltered
    log image — the tones just inside print black (thin negative side for C-41).
    Channel differences here are the residual shadow cast that auto
    shadow-neutral cancels.
    """

    if roi or analysis_buffer > 0:
        sorted_grid = None
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    p = float(EXPOSURE_CONSTANTS["shadow_neutral_percentile"])
    if sorted_grid is not None:
        refs = [float(v) for v in percentile_from_sorted(sorted_grid, p)]
    else:
        img_log = _block_median_grid(img_log)
        refs = [float(np.percentile(img_log[:, :, ch], p)) for ch in range(3)]
    return (refs[0], refs[1], refs[2])


def measure_shadow_log_refs(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Linear-image wrapper around measure_shadow_refs_from_log.
    """
    img_log = to_log_density(image)
    return measure_shadow_refs_from_log(img_log, roi, analysis_buffer)


def _rms_chroma(triplets: np.ndarray) -> np.ndarray:
    """Distance from the neutral axis in normalized-density space (pairwise RMS):
    rotation-symmetric around grey, so the near-neutral ranking is hue-uniform
    (max-min scores an opposed R/B split double a same-side deviation)."""
    r, g, b = triplets[..., 0], triplets[..., 1], triplets[..., 2]
    return np.sqrt(((r - g) ** 2 + (g - b) ** 2 + (r - b) ** 2) / 3.0)


def measure_neutral_axis_from_log(
    img_log: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """
    Per-channel neutral axis: median raw-log density at a highlight, a midtone and a shadow
    luma band, over each band's lowest-chroma pixels. Two passes: pass 1 selects through the
    residual cast under a loose cap (a strong but correctable cast must not collapse the axis;
    saturated content still fails it), then the affine R/B->G correction implied by its
    mid+shadow refs re-ranks chroma so pass 2 selects true neutrals under the strict cap.
    Returns (midtone, shadow, highlight, confidence) — highlight is None when that band has no
    trustworthy neutral set (callers then fit a 2-point line); confidence in [0,1] combines the
    grey sets' corrected tightness, the midtone sample size and mid<->shadow deviation agreement
    (drives Auto Cast Removal). None overall when midtone or shadow is missing (shadow tie).
    """

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)
    norm = normalize_log_image(img_log, bounds)
    luma = LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2]

    flat_log = img_log.reshape(-1, 3)
    norm_f = norm.reshape(-1, 3)
    luma_f = luma.reshape(-1)
    chroma_f = _rms_chroma(norm_f)

    c = EXPOSURE_CONSTANTS
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    pass1_cap = float(c["neutral_axis_first_pass_cap"])
    min_px = int(c["neutral_axis_min_pixels"])
    epsilon = 1e-6

    def _band_refs(
        lo: float, hi: float, chroma_vals: np.ndarray, cap_val: float
    ) -> Optional[Tuple[Tuple[float, float, float], float, int]]:
        band = (luma_f >= lo) & (luma_f <= hi)
        if int(band.sum()) < min_px:
            return None
        band_chroma = chroma_vals[band]
        thr = float(np.quantile(band_chroma, q))
        keep = band_chroma <= thr
        idx = np.nonzero(band)[0][keep]
        near_neutral_chroma = float(np.median(band_chroma[keep])) if idx.size else cap_val
        if idx.size < min_px or near_neutral_chroma > cap_val:
            return None
        # One gather for all three channels: the per-channel fancy index is the cost here.
        sel = flat_log[idx]
        refs = (float(np.median(sel[:, 0])), float(np.median(sel[:, 1])), float(np.median(sel[:, 2])))
        return (refs, near_neutral_chroma, int(idx.size))

    def _norm_ref(refs: Tuple[float, float, float]) -> Tuple[float, float, float]:
        out = []
        for ch in range(3):
            denom = bounds.ceils[ch] - bounds.floors[ch]
            if abs(denom) < epsilon:
                denom = epsilon if denom >= 0 else -epsilon
            out.append((refs[ch] - bounds.floors[ch]) / denom)
        return (out[0], out[1], out[2])

    hb = c["neutral_axis_highlight_band"]
    mb = c["neutral_axis_mid_band"]
    sb = c["neutral_axis_shadow_band"]
    mid1 = _band_refs(float(mb[0]), float(mb[1]), chroma_f, pass1_cap)
    sh1 = _band_refs(float(sb[0]), float(sb[1]), chroma_f, pass1_cap)
    if mid1 is None or sh1 is None:
        return None

    nm, ns = _norm_ref(mid1[0]), _norm_ref(sh1[0])
    corrected = norm_f.copy()
    for ch in (0, 2):
        du = nm[ch] - ns[ch]
        if abs(du) < epsilon:
            a, b = 1.0, nm[1] - nm[ch]
        else:
            a = (nm[1] - ns[1]) / du
            b = nm[1] - a * nm[ch]
        corrected[:, ch] = a * norm_f[:, ch] + b
    chroma2_f = _rms_chroma(corrected)

    mid = _band_refs(float(mb[0]), float(mb[1]), chroma2_f, cap)
    shadow = _band_refs(float(sb[0]), float(sb[1]), chroma2_f, cap)
    if mid is None or shadow is None:
        return None
    highlight = _band_refs(float(hb[0]), float(hb[1]), chroma2_f, cap)

    # Confidence: the corrected tightness of the grey sets, times the midtone sample size,
    # times the mid/shadow deviation agreement, where a dead zone passes plausible crossover.
    n0 = float(c["neutral_axis_confidence_n0"])
    dead = float(c["neutral_axis_agreement_deadzone"])
    scale = float(c["neutral_axis_agreement_scale"])
    tight = float(np.clip(1.0 - max(mid[1], shadow[1]) / cap, 0.0, 1.0))
    size_term = mid[2] / (mid[2] + n0)
    dm, ds = _norm_ref(mid[0]), _norm_ref(shadow[0])
    spread = max(abs((dm[ch] - dm[1]) - (ds[ch] - ds[1])) for ch in (0, 2))
    agree = 1.0 - min(max(spread - dead, 0.0) / scale, 1.0)
    confidence = float(np.clip(tight * size_term * agree, 0.0, 1.0))
    return (mid[0], shadow[0], highlight[0] if highlight is not None else None, confidence)


def measure_neutral_axis(
    image: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
):
    """Linear-image wrapper around measure_neutral_axis_from_log."""
    img_log = to_log_density(image)
    return measure_neutral_axis_from_log(img_log, bounds, roi, analysis_buffer)


def luminance_density_range(bounds: LogNegativeBounds) -> float:
    """
    Single global density range as a Rec.709 luminance weighting of the
    per-channel ranges. Replaces the green-only range so frames with a strong
    single-channel cast don't swing the slope as hard, while green still
    dominates so calibrated grade behaviour barely shifts. abs() keeps it
    sign-safe for a slide's reversed (f > c) window.
    """
    rr = abs(bounds.ceils[0] - bounds.floors[0])
    rg = abs(bounds.ceils[1] - bounds.floors[1])
    rb = abs(bounds.ceils[2] - bounds.floors[2])
    return float(LUMA_R * rr + LUMA_G * rg + LUMA_B * rb)


def _textured_cells(lum: np.ndarray) -> np.ndarray:
    """
    Cells of the prefiltered luma grid that carry detail. The grid is tiled into
    sectors of 2x2 blocks of activity_block cells; a sector votes when its four block
    means span more than activity_gate_density (Boyack & Juenger, US 5,724,456). Block
    means average grain out, and a flat sector beside a textured one stays out because
    only its own blocks are compared. Falls back to every cell when too few pass.
    """

    c = EXPOSURE_CONSTANTS
    b = int(c["activity_block"])
    h, w = lum.shape
    hs, ws = (h // (2 * b)) * 2 * b, (w // (2 * b)) * 2 * b
    if hs == 0 or ws == 0:
        return lum.ravel()
    core = lum[:hs, :ws]
    blocks = core.reshape(hs // b, b, ws // b, b).mean(axis=(1, 3))
    sectors = blocks.reshape(hs // (2 * b), 2, ws // (2 * b), 2)
    active = (sectors.max(axis=(1, 3)) - sectors.min(axis=(1, 3))) > float(c["activity_gate_density"])
    if active.mean() < float(c["activity_min_fraction"]):
        return lum.ravel()
    mask = np.repeat(np.repeat(active, 2 * b, axis=0), 2 * b, axis=1)
    return core[mask]


def _textured_norm_luma(grid: ImageBuffer, bounds: LogNegativeBounds) -> np.ndarray:
    """Luma of the grid normalized against `bounds`, textured cells only."""
    epsilon = 1e-6
    norm = np.empty_like(grid)
    for ch in range(3):
        f = bounds.floors[ch]
        denom = bounds.ceils[ch] - f
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, :, ch] = (grid[:, :, ch] - f) / denom
    return _textured_cells(LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2])


def measure_shadow_point_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """Normalized luma of the textured dark tail (shadow_reach_percentile), the tone
    Auto Grade's shadow reach prints at paper black."""

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    lum = _textured_norm_luma(_block_median_grid(img_log), bounds)
    return float(np.percentile(lum, float(EXPOSURE_CONSTANTS["shadow_reach_percentile"])))


def measure_highlight_point_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """Normalized luma of the textured bright tail (highlight_hold_percentile), the tone
    Auto Grade's highlight hold keeps off paper white."""

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    lum = _textured_norm_luma(_block_median_grid(img_log), bounds)
    return float(np.percentile(lum, float(EXPOSURE_CONSTANTS["highlight_hold_percentile"])))


def measure_anchor_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    assumed: Optional[float] = None,
) -> float:
    """
    Per-frame exposure anchor: where this negative's midtone sits in [0, 1],
    replacing the fixed assumed_anchor. Read over the textured cells of the
    block-median grid, as the mean of the trimmed window and its midpoint averaged
    (Boyack & Juenger, US 5,724,456), so a skewed histogram is placed by its
    detail-bearing span rather than by its median.

    Partial metering: the anchor moves only anchor_meter_strength of the way from
    `assumed` toward the metered median, so a deliberately low-key (dark) or
    high-key (bright) scene keeps most of its intended key instead of being
    forced to mid-gray, while gross mis-exposure is still pulled toward correct.
    A linear pull (no key-dependent amplification) keeps it predictable. Finally
    clamped to `assumed` +/- anchor_meter_band as a hard safety guard.

    `assumed` defaults to EXPOSURE_CONSTANTS["assumed_anchor"], calibrated for the
    paper path's own per-frame-adaptive bounds. The transfer path's `bounds` are
    fixed rather than adaptive, so its expected anchor position is a different
    number (transfer_assumed_anchor) in the same [0, 1] units; pass it explicitly.
    """

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    lum = _textured_norm_luma(_block_median_grid(img_log), bounds)
    clip = float(EXPOSURE_CONSTANTS["anchor_trim_clip"])
    lo, hi = np.percentile(lum, [clip, 100.0 - clip])
    inner = lum[(lum >= lo) & (lum <= hi)]
    measured = 0.5 * (float(inner.mean()) + 0.5 * (float(lo) + float(hi)))

    a = float(EXPOSURE_CONSTANTS["assumed_anchor"]) if assumed is None else float(assumed)
    strength = float(EXPOSURE_CONSTANTS["anchor_meter_strength"])
    band = float(EXPOSURE_CONSTANTS["anchor_meter_band"])
    anchor = a + strength * (measured - a)
    return float(min(max(anchor, a - band), a + band))


def measure_anchor(
    image: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_anchor_from_log.
    """
    img_log = to_log_density(image)
    return measure_anchor_from_log(img_log, bounds, roi, analysis_buffer)


def measure_textural_range_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Per-frame textural density range: the P10-P90 luminance spread of the
    prefiltered log image, in log10-density units. This is the *useful* scene
    range that grade selection fits to paper — block-median prefiltering and the
    inner percentiles reject speculars / film-base / dust, so it is far more
    outlier-robust than the floor-to-ceil extreme range.
    """

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    lum = _textured_cells(LUMA_R * img_log[:, :, 0] + LUMA_G * img_log[:, :, 1] + LUMA_B * img_log[:, :, 2])
    clip = float(EXPOSURE_CONSTANTS["textural_range_clip"])
    lo, hi = np.percentile(lum, [clip, 100.0 - clip])
    return float(abs(hi - lo))


def measure_textural_range(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_textural_range_from_log.
    """
    img_log = to_log_density(image)
    return measure_textural_range_from_log(img_log, roi, analysis_buffer)


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """Stretches log-data to fit [0, 1]."""
    floors = np.array(bounds.floors, dtype=np.float32)
    ceils = np.array(bounds.ceils, dtype=np.float32)
    out = _normalize_log_image(img_log.astype(np.float32, copy=False), floors, ceils)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def _sample_log_bounds(
    img_log: np.ndarray,
    percentile_clip: float,
    base: float,
    sorted_grid: Optional[np.ndarray] = None,
) -> tuple[list, list]:
    """
    Per-channel (floors, ceils) at one clip level. `base` is the robust baseline
    clip added on top of the slider value; negative slider values expand outward
    by a log-density margin instead.
    """
    if percentile_clip >= 0:
        clip = max(0.00001, min(50.0, percentile_clip + base))
        margin = 0.0
    else:
        # Margin mode expands from the same robust basis, so the slider stays continuous through
        # its neutral position.
        clip = base
        margin = -percentile_clip
    p_low, p_high = np.float64(clip), np.float64(100.0 - clip)

    def _pct(p) -> list:
        if sorted_grid is not None:
            return [float(v) for v in percentile_from_sorted(sorted_grid, p)]
        return [float(np.percentile(img_log[:, :, ch], p)) for ch in range(3)]

    floors = _pct(p_low)
    ceils = _pct(p_high)

    if margin > 0.0:
        # Expand outward; per-channel sign handles both f < c and f > c.
        for ch in range(3):
            if ceils[ch] >= floors[ch]:
                floors[ch] -= margin
                ceils[ch] += margin
            else:
                floors[ch] += margin
                ceils[ch] -= margin

    return floors, ceils


def _same_pixel_color_floor_refs(
    img_log: ImageBuffer,
    luma_floors: list,
    luma_ceils: list,
    base_refs: Tuple[float, float, float],
    color_clip: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Dense-end (print-white) color refs from one shared pixel set: the luma-extreme
    band's lowest-chroma subset, chroma measured base-anchored (offsets from the
    thin-end refs, per-channel span as provisional gamma, refined once from the
    band medians). Independent per-channel percentiles read a different scene
    object per channel, so colored highlight content masquerades as film cast;
    a shared chroma-gated set cannot. The thin end needs no such treatment —
    density on real film is bounded below by base, anchoring per-channel ceils.
    None when the band's neutral set is too small or too chromatic (caller falls
    back to the percentile pass).
    """

    c = EXPOSURE_CONSTANTS
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    min_px = int(c["neutral_axis_min_pixels"])
    width = float(c["color_bounds_band_width"])
    epsilon = 1e-6

    flat = img_log.reshape(-1, 3).astype(np.float64)
    base = np.asarray(base_refs, dtype=np.float64)
    norm = np.empty_like(flat)
    for ch in range(3):
        denom = luma_ceils[ch] - luma_floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, ch] = (flat[:, ch] - luma_floors[ch]) / denom
    luma = LUMA_R * norm[:, 0] + LUMA_G * norm[:, 1] + LUMA_B * norm[:, 2]

    clip = max(0.00001, min(50.0 - width, float(color_clip)))
    lo, hi = np.percentile(luma, [clip, clip + width])
    band = (luma >= lo) & (luma <= hi)
    if int(band.sum()) < min_px:
        return None
    d = flat[band] - base[None, :]

    def _select(gamma: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        g = gamma.copy()
        g[np.abs(g) < epsilon] = epsilon
        chroma = _rms_chroma(d / g[None, :])
        thr = float(np.quantile(chroma, q))
        sel = chroma <= thr
        if int(sel.sum()) < min_px:
            return None
        return sel, float(np.median(chroma[sel]))

    spans = np.array([luma_floors[ch] - base_refs[ch] for ch in range(3)], dtype=np.float64)
    first = _select(spans)
    # Pass-1 loose cap: a homogeneous colored cluster would otherwise be self-normalized to
    # zero chroma by pass 2 and read as neutral.
    if first is None or first[1] > float(c["neutral_axis_first_pass_cap"]):
        return None
    provisional = np.median(d[first[0]], axis=0)
    if np.any(np.abs(provisional) < epsilon):
        return None
    second = _select(provisional)
    if second is None or second[1] > cap:
        return None
    refs = base + np.median(d[second[0]], axis=0)
    return (float(refs[0]), float(refs[1]), float(refs[2]))


def analyze_log_exposure_bounds(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
    unmix: Optional[np.ndarray] = None,
    thin_end_override: Optional[Tuple[float, float, float]] = None,
) -> LogNegativeBounds:
    """
    Performs full analysis pass on a linear image to find density floors/ceils.

    Two independent axes are sampled and recombined:
      - percentile_clip (luma): drives the overall black/white-point luminance and
        span (ceil-floor) — i.e. dynamic range / highlight headroom. Sampled at the
        gentle base_luma_clip baseline; slider semantics are:
          > 0  clips the histogram tails (added on top of the baseline clip).
          = 0  robust extremes (block-median prefilter + baseline clip).
          < 0  outward headroom: bounds pushed BEYOND the robust extremes by the margin.
      - color_clip (color): the absolute per-tail clip percentile for the per-channel
        color deviation (white balance / orange-mask cast). A tighter (larger) clip
        gives a more robust channel balance; a gentler (smaller) clip samples nearer
        the extremes. Default neutral is base_color_clip.
    The luminance centre+span comes from the luma sampling, the per-channel color
    offsets from the color sampling, so the cast clip is tunable without compressing
    highlights. Identical channels (mono) give zero deviation at any clip.
    """
    img_log = to_log_density(image)
    img_log = unmix_log_image(img_log, unmix)
    return analyze_log_exposure_bounds_from_log(
        img_log, roi, analysis_buffer, percentile_clip, color_clip, thin_end_override=thin_end_override
    )


def analyze_log_exposure_bounds_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
    sorted_grid: Optional[np.ndarray] = None,
    thin_end_override: Optional[Tuple[float, float, float]] = None,
) -> LogNegativeBounds:
    """Log-image core of analyze_log_exposure_bounds (skips the log10).

    ``sorted_grid`` is `sorted_channel_grid(img_log)` from a caller that already holds it;
    it stands in for both percentile passes, so it is only valid when no ROI or analysis
    buffer is left to apply here.
    """

    if roi or analysis_buffer > 0:
        sorted_grid = None
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]

    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    base_luma = float(EXPOSURE_CONSTANTS["base_luma_clip"])

    floors, ceils = _sample_log_bounds(img_log, percentile_clip, base_luma, sorted_grid)

    # Color pass: per-channel deviations recombined onto the luma mean centre and span. The
    # ceils (thin end, base-anchored) come from per-channel percentiles at color_clip. The
    # floors (dense end, scene content) prefer the same-pixel chroma-gated band refs and fall
    # back to the percentile pass when the band holds no trustworthy neutrals, and always for
    # margin-mode clips.
    c_floors, c_ceils = _sample_log_bounds(img_log, color_clip, 0.0, sorted_grid)
    if thin_end_override is not None:
        # Manual film base: the thin end is the measured base density, which also anchors
        # the dense-end same-pixel refs.
        c_ceils = [float(v) for v in thin_end_override]
    if color_clip >= 0:
        sp = _same_pixel_color_floor_refs(img_log, floors, ceils, (c_ceils[0], c_ceils[1], c_ceils[2]), color_clip)
        if sp is not None:
            c_floors = [sp[0], sp[1], sp[2]]
    mean_lf, mean_lc = sum(floors) / 3.0, sum(ceils) / 3.0
    mean_cf, mean_cc = sorted(c_floors)[1], sorted(c_ceils)[1]
    floors = [mean_lf + (c_floors[ch] - mean_cf) for ch in range(3)]
    ceils = [mean_lc + (c_ceils[ch] - mean_cc) for ch in range(3)]

    if thin_end_override is not None:
        ceils = [float(v) for v in thin_end_override]
    return LogNegativeBounds(
        (floors[0], floors[1], floors[2]),
        (ceils[0], ceils[1], ceils[2]),
    )


