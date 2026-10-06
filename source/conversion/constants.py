"""Constants ported from NegPy (GPL-3.0). See NOTICE.md."""
from typing import Any, Dict

EXPOSURE_CONSTANTS: Dict[str, Any] = {
    # Max absolute density offset per CMY white-balance slider unit.
    "cmy_max_density": 0.2,
    # Scales the density slider's effect on the exposure pivot.
    "density_multiplier": 0.2,
    # Density the reference tone (assumed_anchor) prints at.
    "anchor_target_density": 0.75,
    # Zone Density (ΔD) weights: mid-sparing sigmoids centred in the three-quarter/
    # quarter tones (offsets from anchor_target_density), so midtones get neither
    # offset. Mirrored as literals in exposure.wgsl: change both together.
    "zone_density_sharpness": 4.0,
    "zone_density_shadow_offset": 0.75,
    "zone_density_highlight_offset": -0.40,
    # Default normalized midtone reference in [0,1] log space (auto_exposure off).
    "assumed_anchor": 0.46,
    # ISO R bounds: hardest and softest grade allowed.
    "iso_r_min": 50.0,
    "iso_r_max": 180.0,
    # Bounds on the per-channel straight-line slope k.
    "slope_min": 2.0,
    "slope_max": 10.0,
    # Physical paper black (D_max) and paper white (D_min) densities.
    "d_max": 2.3,
    "d_min": 0.06,
    # Global multiplier on the toe and shoulder slider values.
    "toe_shoulder_strength": 0.85,
    # Asymmetric H&D print curve: a straight midtone of slope k between a toe
    # (shadow roll-off to d_max) and a shoulder (highlight roll-off to d_min),
    # each a smooth softplus bound. The sliders set roll-off *height*; the
    # *_sharpness_base / width set sharpness.
    # a_sh = toe_sharpness_base * width_ref / toe_width (shoulder is the same).
    "toe_sharpness_base": 6.0,
    "shoulder_sharpness_base": 3.0,
    # Reference width that normalises both sharpness coefficients.
    "toeshoulder_width_ref": 2.5,
    # Density moved per slider unit: d_max_eff = d_max - toe*this,
    # d_min_eff = d_min + shoulder*this. toe_height is larger because density is
    # log10, so a ΔD near d_max reads smaller in L* than the same ΔD near d_min.
    "toe_height": 0.90,
    "shoulder_height": 0.35,
    # k = grade_contrast_scale * density_range / (ISO_R/100). Calibrated so R115
    # reproduces the legacy mid-curve slope.
    "grade_contrast_scale": 2.9,
    # Side length of the block-median pre-filter grid for exposure analysis.
    "analysis_grid": 1024,
    # Base percentile clip on the luma-range histogram analysis.
    "base_luma_clip": 0.01,
    # Neutral per-tail clip for per-channel balance (orange-mask cast removal),
    # independent of luma range. The slider spans percentiles around this.
    "base_color_clip": 1.0,
    # Percentile sampled as the per-channel shadow reference for cast detection.
    "shadow_neutral_percentile": 98.0,
    # Scan-exposure warning: linear level treated as sensor-white clipping. Film
    # base and scene shadows sit near sensor white, so clipped pixels collapse
    # distinct densities to D=0.
    "scan_clip_level": 0.99,
    # Per-channel clipped fraction above which the Analysis panel warns.
    "scan_clip_warn": 0.01,
    # Max normalized shadow cast (green - channel) that Cast Removal corrects.
    "cast_removal_max_offset": 0.1,
    # Cast Removal neutral axis: per-channel refs at a highlight/midtone/shadow luma
    # band, each over the band's lowest-chroma pixels. R/B fit green's axis with a
    # quadratic through all three, else a line through mid+shadow. Bands are
    # normalized luma.
    "neutral_axis_highlight_band": (0.10, 0.30),
    "neutral_axis_mid_band": (0.40, 0.60),
    "neutral_axis_shadow_band": (0.72, 0.92),
    # Lowest-chroma fraction of each band kept as the near-neutral set.
    "neutral_axis_chroma_quantile": 0.30,
    # Above this median corrected chroma (pass 2) the set is not trustworthy:
    # fall back to the shadow-only tie.
    "neutral_axis_chroma_cap": 0.29,
    # Pass-1 chroma ceiling: admits strong correctable casts, rejects saturated content.
    "neutral_axis_first_pass_cap": 0.55,
    "neutral_axis_min_pixels": 64,
    # Confidence sample-size half-point: the size term is n / (n + this).
    "neutral_axis_confidence_n0": 256,
    # Mid/shadow deviation-difference dead zone (a plausible crossover passes free)
    # and the roll-off width of the confidence agreement term beyond it.
    "neutral_axis_agreement_deadzone": 0.10,
    "neutral_axis_agreement_scale": 0.20,
    # Width (percentile points) of the luma-extreme band the same-pixel color
    # floor refs read; Color Clip sets the band's depth.
    "color_bounds_band_width": 4.0,
    # Clamp on each channel's deviation from green at any anchor.
    "midtone_cast_max_offset": 0.2,
    # Curvature clamp (fraction of slope, <0.5): keeps the per-channel core
    # monotonic on [0,1].
    "neutral_axis_curv_max_ratio": 0.45,
    # Gain clamp for the transparency-path affine cast solve (and its reciprocal):
    # bounds the correction when green's midtone and shadow refs sit close together.
    "cast_affine_gain_limit": 2.0,
    # Activity gate shared by the anchor and textural-range meters: the grid is tiled
    # into sectors of 2x2 blocks of activity_block cells, and a sector votes only if
    # its block means span more than this log10 density (about a third of a stop on a
    # colour negative), so rebate, sky and walls drop out. Below activity_min_fraction
    # of sectors passing, every cell votes.
    "activity_block": 8,
    "activity_gate_density": 0.05,
    "activity_min_fraction": 0.05,
    # Per-tail percentile trim of the textured-cell window the anchor is placed from.
    "anchor_trim_clip": 5.0,
    # Safety band around assumed_anchor that clamps the auto-metered result.
    "anchor_meter_band": 0.12,
    # Auto Density: fraction of the distance from assumed_anchor toward the
    # metered anchor that is applied.
    "anchor_meter_strength": 0.2,
    # Grade-coupled baseline roll-off: toe_eff += this * slope_norm, so hard
    # grades get more toe. The 0.35/0.90 factor holds the baseline ΔD
    # (this * toe_height) constant against the perceptual toe_height.
    "toe_grade_strength": 0.15 * 0.35 / 0.90,
    "shoulder_grade_strength": 0.12,
    # Auto Grade: effective_range = K * floor_ceil * ((1 - s) + s * nominal / textural).
    "auto_grade_target": 0.85,
    # Auto Grade adaptation strength s: 0 = fixed paper, 1 = every frame's textural
    # range prints alike. Alkofer (US 4,731,671) corrects 60-80% toward the norm.
    "auto_grade_strength": 0.4,
    # Cap on the textural range's print span, as a multiple of a normal negative's.
    "auto_grade_max_overfill": 1.2,
    # Textural (P10-P90) density range of a normal negative, log10 D: the norm the
    # grade is shrunk toward.
    "auto_grade_nominal_range": 0.9,
    # floor_ceil/textural ratio of a normal negative; sizes the unmetered fallback.
    "auto_grade_nominal_ratio": 1.5,
    # Percentile margin for the "textural" scene range (rejects speculars and dust).
    "textural_range_clip": 10.0,
    # Auto Grade shadow reach: the dark tail of the textured cells (this percentile of
    # the normalized luma) must print at least this straight-line density; the grade
    # only ever goes harder for it (Gindele, US 7,113,649).
    "shadow_reach_percentile": 99.0,
    "shadow_reach_density": 1.9,
    # Auto Grade highlight hold, the soft-exposure half of a split-grade print: the bright
    # tail of the textured cells (this percentile) must print at least this density, met by
    # an automatic highlight zone burn (never a lift), capped at highlight_hold_max. 0 = off.
    "highlight_hold_percentile": 2.0,
    "highlight_hold_density": 0.10,
    "highlight_hold_max": 0.5,
    # Preflash threshold: the density above paper white that the ISO R range starts at.
    "preflash_threshold_density": 0.04,
    # Flat / digital-intermediate master (RenderIntent.FLAT). A log-video master:
    # the normalized log signal becomes the code value directly, with no 10^-D
    # decode and no sRGB OETF, so it stays flat and fully invertible.
    # code = clip(flat_log_lift + flat_log_gain*(1 - val), 0, 1). Fixed, with no
    # per-frame metering, so a roll of equal scans renders identically.
    # Log-master contrast; <1 keeps it flat.
    "flat_log_gain": 0.65,
    # Code value the scene shadow (val=1) lands on.
    "flat_log_lift": 0.10,
    # Variable-gamma paper S-curve. Extra local gamma at the midtone centre via
    # v += gamma*width*tanh((v - v_star)/width), easing to zero toward the toe and
    # shoulder, like a real paper curve. Anchor-preserving.
    # Extra midtone gamma at the centre (0 disables the S-shape).
    "paper_midtone_gamma": 0.05,
    # Density half-width over which that boost eases to the tails.
    "paper_gamma_width": 0.6,
    # Chroma (RMS dye-density spread above paper base) that Separation Damping
    # leaves untouched: below it a pixel takes the full Dye Separation push, above
    # it the inverse. Both ends are failure modes: too high and every pixel takes
    # the same push (the frame-wide matrix again), too low and the crossing leaves
    # the picture. Inlined as a WGSL literal: change both.
    "separation_damping_ref_spread": 0.35,
}


# Transfer (slide) path constants
TRANSFER_DENSITY_RANGE = 3.0
SEPARATION_CAP_LEVEL = 2.5
SEPARATION_CAP_SPREAD = 1.5
SEPARATION_CAP_SOFTNESS = 0.15

TRANSFER_CONSTANTS = {
    # Reference grade (ISO R) that means "no contrast change". MUST equal the grade the app
    # ships in DEFAULT_WORKSPACE_CONFIG, or the transfer stops being identity at defaults.
    # Mirrored rather than imported to keep this module dependency-free;
    # test_transparency_transfer.py asserts the two agree.
    "transfer_grade_ref": 115.0,
    # Stops of exposure per unit of the density slider (higher density = darker).
    "transfer_density_stops": 2.0,
    # Contrast pivot as a density: mid-grey at ~18% of the white level.
    "transfer_contrast_pivot": 0.75,
    # Knees, in density, where the roll-offs start biting. The toe works down from the shadow
    # end and the shoulder up from the highlight end.
    "transfer_toe_knee": 1.6,
    "transfer_shoulder_knee": 0.35,
    # Baseline exposure, in stops, applied before the display rendering below. The decode
    # anchors the signal to the SENSOR WHITE LEVEL (no_auto_bright, adjust_maximum_thr=0), so
    # a frame exposed below clipping arrives correspondingly dark, while every consumer raw
    # converter opens from a baseline instead. 0.7 EV is darktable's shipped default for raw
    # files, a published camera-agnostic figure rather than one fitted to any rig. Fixed,
    # never metered: metering would make a bracketed set converge again.
    "transfer_baseline_ev": 0.7,
    # Softness of both knees, in density, at the reference width. Larger blends over a wider
    # tonal span. The Toe and Shoulder Width sliders scale this about their own default, so
    # the reference must equal ExposureConfig.toe_width / shoulder_width.
    "transfer_knee_width": 0.45,
    "transfer_width_ref": 2.5,
}


# ---- Color constants (appended by output/transfer modules) -------------------
import numpy as _np  # noqa: E402

LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722  # Rec.709 luma

# Adobe RGB (1998), D65 white
XYZ_TO_ADOBE_RGB = _np.array(
    [[2.0413690, -0.5649464, -0.3446944],
     [-0.9692660, 1.8760108, 0.0415560],
     [0.0134474, -0.1183897, 1.0154096]], dtype=_np.float64)
ADOBE_RGB_TO_XYZ = _np.linalg.inv(XYZ_TO_ADOBE_RGB)

# sRGB (IEC 61966-2-1), D65 white
XYZ_TO_SRGB = _np.array(
    [[3.2404542, -1.5371385, -0.4985314],
     [-0.9692660, 1.8760108, 0.0415560],
     [0.0556434, -0.2040259, 1.0572252]], dtype=_np.float64)
SRGB_TO_XYZ = _np.linalg.inv(XYZ_TO_SRGB)

ADOBE_RGB_TO_SRGB = XYZ_TO_SRGB @ ADOBE_RGB_TO_XYZ

# Adobe RGB TRC: pure 563/256 power
WORKING_GAMMA = 563.0 / 256.0
