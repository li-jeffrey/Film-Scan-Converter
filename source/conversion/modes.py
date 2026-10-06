"""Process-mode helpers ported from NegPy features/process/models.py (GPL-3.0)."""
from enum import StrEnum
from typing import Tuple

DEFAULT_CAST_REMOVAL = 1.0


class ProcessMode(StrEnum):
    C41 = "Color Negative"
    BW = "B&W Negative"
    E6 = "Transparency"

    @classmethod
    def _missing_(cls, value: object) -> "ProcessMode":
        return {"C41": cls.C41, "B&W": cls.BW, "E-6": cls.E6}.get(str(value), cls.C41)


def cast_removal_for_mode(mode: str, strength: float) -> float:
    """Cast Removal strength `mode` starts at; a user-chosen strength survives a mode switch."""
    if mode == ProcessMode.E6:
        return 0.0 if strength == DEFAULT_CAST_REMOVAL else strength
    return DEFAULT_CAST_REMOVAL if strength == 0.0 else strength


def auto_meter_for_mode(mode: str, current: bool) -> bool:
    """Auto Density/Auto Grade toggle after a switch into `mode` (negatives on, slides off)."""
    if mode == ProcessMode.E6:
        return False if current else current
    return True if not current else current


def per_channel_point_offsets(
    white_point_offset: float = 0.0, black_point_offset: float = 0.0
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Per-channel (white, black) point offsets. Global only (per-layer trims dropped)."""
    w, b = float(white_point_offset), float(black_point_offset)
    return (w, w, w), (b, b, b)
