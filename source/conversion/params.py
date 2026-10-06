"""Conversion parameters, slider mappings and the GUI spec table.

Ported in spirit from NegPy's ExposureConfig/ProcessConfig (GPL-3.0); only the subset
Film Scan Converter exposes is kept.
"""
import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import EXPOSURE_CONSTANTS
from .modes import ProcessMode, auto_meter_for_mode, cast_removal_for_mode

SETTINGS_VERSION = 2

# FSC film_type -> ProcessMode (3 = crop only has no conversion)
FILM_TYPE_BW, FILM_TYPE_COLOUR, FILM_TYPE_SLIDE, FILM_TYPE_CROP = 0, 1, 2, 3
_MODE_FOR_FILM_TYPE = {0: ProcessMode.BW, 1: ProcessMode.C41, 2: ProcessMode.E6}

# ---- slider <-> value maps (NegPy desktop/view/sidebar/process.py) ----
_LUMA_MARGIN_MIN, _LUMA_MARGIN_MAX = 1e-6, 1.0
_COLOR_CLIP_NEUTRAL = float(EXPOSURE_CONSTANTS["base_color_clip"])
_COLOR_CLIP_MIN, _COLOR_CLIP_MAX = 1e-6, 5.0


def luma_clip_to_value(pos: float) -> float:
    if pos >= 0:
        return math.pow(10, 0.05 * pos - 5)
    lo, hi = math.log10(_LUMA_MARGIN_MIN), math.log10(_LUMA_MARGIN_MAX)
    return -math.pow(10, lo + (-pos / 100.0) * (hi - lo))


def luma_clip_to_slider(v: float) -> float:
    if v >= 0:
        return 20 * (math.log10(max(v, 1e-5)) + 5)
    lo, hi = math.log10(_LUMA_MARGIN_MIN), math.log10(_LUMA_MARGIN_MAX)
    return -100.0 * (math.log10(-v) - lo) / (hi - lo)


def color_clip_to_value(pos: float) -> float:
    ln = math.log10(_COLOR_CLIP_NEUTRAL)
    end = math.log10(_COLOR_CLIP_MAX if pos >= 0 else _COLOR_CLIP_MIN)
    return math.pow(10, ln + (abs(pos) / 100.0) * (end - ln))


def color_clip_to_slider(v: float) -> float:
    ln = math.log10(_COLOR_CLIP_NEUTRAL)
    lv = math.log10(min(max(v, _COLOR_CLIP_MIN), _COLOR_CLIP_MAX))
    if v >= _COLOR_CLIP_NEUTRAL:
        return 100.0 * (lv - ln) / (math.log10(_COLOR_CLIP_MAX) - ln)
    return -100.0 * (lv - ln) / (math.log10(_COLOR_CLIP_MIN) - ln)


@dataclass(frozen=True)
class ConversionParams:
    film_type: int = FILM_TYPE_COLOUR
    # print
    density: float = 1.0
    grade: float = 115.0
    auto_density: bool = True
    auto_grade: bool = True
    toe: float = 0.0
    shoulder: float = 0.0
    toe_width: float = 2.5
    shoulder_width: float = 2.5
    shadow_density: float = 0.0
    highlight_density: float = 0.0
    paper_profile: str = "neutral"
    paper_dmin: bool = False
    paper_black: bool = False
    # colour
    wb_cyan: float = 0.0
    wb_magenta: float = 0.0
    wb_yellow: float = 0.0
    dye_separation: float = 1.0
    cast_removal: float = 1.0
    # crosstalk
    crosstalk_profile: str = "Generic C41"
    crosstalk_strength: float = 0.0
    # metering
    white_point_offset: float = 0.0
    black_point_offset: float = 0.0
    luma_clip_pos: float = 0.0
    color_clip_pos: float = 0.0
    analysis_border: Tuple[float, float] = (5.0, 5.0)  # percent of crop (x, y)
    base_rgb: Optional[Tuple[float, float, float]] = None  # linear film base, None = Auto
    # io
    output_colour_space: str = "sRGB"

    @property
    def mode(self) -> ProcessMode:
        return _MODE_FOR_FILM_TYPE.get(int(self.film_type), ProcessMode.C41)

    @property
    def luma_clip(self) -> float:
        return luma_clip_to_value(self.luma_clip_pos)

    @property
    def color_clip(self) -> float:
        return color_clip_to_value(self.color_clip_pos)

    # cache keys -----------------------------------------------------------------
    def grid_key(self) -> tuple:
        return (self.film_type, self.crosstalk_profile, round(self.crosstalk_strength, 6),
                tuple(self.analysis_border))

    def analysis_key(self) -> tuple:
        return self.grid_key() + (round(self.luma_clip_pos, 6), round(self.color_clip_pos, 6),
                                  self.base_rgb)

    def norm_key(self) -> tuple:
        return self.analysis_key() + (round(self.white_point_offset, 6), round(self.black_point_offset, 6))


PARAM_KEYS = tuple(f.name for f in fields(ConversionParams))


def from_photo(photo: Any) -> ConversionParams:
    """Build params from any object (RawProcessing) carrying the per-photo attributes."""
    kw: Dict[str, Any] = {}
    for k in PARAM_KEYS:
        if k == "output_colour_space":
            continue
        if hasattr(photo, k):
            v = getattr(photo, k)
            if k == "analysis_border":
                v = tuple(v)
            elif k == "base_rgb" and v is not None:
                v = tuple(float(x) for x in np.asarray(v, dtype=np.float64).ravel()[:3]) if _is_linear_base(v) else None
            kw[k] = v
    cls_params = getattr(photo, "class_parameters", {}) or {}
    kw["output_colour_space"] = cls_params.get("output_colour_space", "sRGB")
    if "analysis_border" not in kw and "ignore_border" in cls_params:
        kw["analysis_border"] = tuple(cls_params["ignore_border"])
    return ConversionParams(**kw)


def _is_linear_base(v: Any) -> bool:
    a = np.asarray(v, dtype=np.float64).ravel()
    return a.size >= 3 and bool(np.all(a[:3] >= 0)) and bool(np.all(a[:3] <= 1.0)) and not np.all(a[:3] == 0)


def mode_defaults(film_type: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cast removal / auto toggles after switching to `film_type` (user-chosen values survive)."""
    mode = _MODE_FOR_FILM_TYPE.get(int(film_type), ProcessMode.C41)
    out = dict(params)
    out["cast_removal"] = cast_removal_for_mode(mode, float(params.get("cast_removal", 1.0)))
    out["auto_density"] = auto_meter_for_mode(mode, bool(params.get("auto_density", True)))
    out["auto_grade"] = auto_meter_for_mode(mode, bool(params.get("auto_grade", True)))
    return out


# (key, label, min, max, step, default, widget) -- widget: 'slider' | 'check' | 'combo'
SPECS = (
    ("density", "Print Density", 0.0, 2.0, 0.05, 1.0, "slider"),
    ("grade", "Grade (ISO R)", 50, 180, 1, 115, "slider"),
    ("auto_density", "Auto Density", None, None, None, True, "check"),
    ("auto_grade", "Auto Grade", None, None, None, True, "check"),
    ("toe", "Toe", -1.0, 1.0, 0.05, 0.0, "slider"),
    ("shoulder", "Shoulder", -1.0, 1.0, 0.05, 0.0, "slider"),
    ("toe_width", "Toe Width", 0.1, 5.0, 0.1, 2.5, "slider"),
    ("shoulder_width", "Shoulder Width", 0.1, 5.0, 0.1, 2.5, "slider"),
    ("shadow_density", "Shadows", -0.9, 0.9, 0.01, 0.0, "slider"),
    ("highlight_density", "Highlights", -0.5, 0.5, 0.01, 0.0, "slider"),
    ("paper_profile", "Paper", None, None, None, "neutral", "combo"),
    ("paper_dmin", "Paper White", None, None, None, False, "check"),
    ("paper_black", "Paper Black", None, None, None, False, "check"),
    ("wb_cyan", "Cyan", -1.0, 1.0, 0.01, 0.0, "slider"),
    ("wb_magenta", "Magenta", -1.0, 1.0, 0.01, 0.0, "slider"),
    ("wb_yellow", "Yellow", -1.0, 1.0, 0.01, 0.0, "slider"),
    ("dye_separation", "Saturation", 0.5, 1.5, 0.05, 1.0, "slider"),
    ("cast_removal", "Cast Removal", 0.0, 1.0, 0.05, 1.0, "slider"),
    ("crosstalk_profile", "Film Stock", None, None, None, "Generic C41", "combo"),
    ("crosstalk_strength", "Crosstalk Strength", 0.0, 1.0, 0.05, 0.0, "slider"),
    ("white_point_offset", "White Point", -0.25, 0.25, 0.01, 0.0, "slider"),
    ("black_point_offset", "Black Point", -0.25, 0.25, 0.01, 0.0, "slider"),
    ("luma_clip_pos", "Luma Clip", -100, 100, 1, 0, "slider"),
    ("color_clip_pos", "Colour Clip", -100, 100, 1, 0, "slider"),
)
SPEC_DEFAULTS = {s[0]: s[5] for s in SPECS}

# keys that survive v1 -> v2 migration unchanged
_V1_KEEP = ("dark_threshold", "light_threshold", "border_crop", "flip", "rotation", "film_type",
            "reject", "remove_dust", "base_detect")


def migrate_v1(d: Dict[str, Any]) -> Dict[str, Any]:
    """Old per-photo settings -> v2: keep geometry/film type/flags, reset the rest to defaults."""
    out: Dict[str, Any] = {k: d[k] for k in _V1_KEEP if k in d}
    out.update(SPEC_DEFAULTS)
    if "sat" in d:
        out["dye_separation"] = float(np.clip(float(d["sat"]) / 100.0, 0.5, 1.5))
    out["base_rgb"] = None  # legacy 8-bit base is incompatible with linear base
    out["settings_version"] = SETTINGS_VERSION
    if "film_type" in out:
        out = mode_defaults(out["film_type"], out)
    return out
