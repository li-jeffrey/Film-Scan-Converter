"""Output encoding: linear Adobe-RGB-primaries render -> display/export code values.

Ported from NegPy (GPL-3.0) kernel/image/logic.py (working OETF, sRGB helpers), vectorized
NumPy. The pipeline renders linear data in Adobe RGB (1998) primaries; this module encodes it
for sRGB, Adobe RGB or a gray (gamma 2.2) target.
"""
from typing import Optional

import numpy as np

from .constants import ADOBE_RGB_TO_SRGB, LUMA_B, LUMA_G, LUMA_R, WORKING_GAMMA

OUTPUT_SPACES = ("sRGB", "Adobe RGB", "Gray")
GRAY_GAMMA = 2.2


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB gamma-encoded -> linear light (IEC 61966-2-1)."""
    img = np.asarray(img, dtype=np.float32)
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Linear light -> sRGB OETF, clipped to [0, 1]."""
    x = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def working_oetf_encode(img: np.ndarray) -> np.ndarray:
    """Scene-linear -> code values in [0,1] using the Adobe RGB TRC (x^(256/563))."""
    x = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return np.power(x, np.float32(1.0 / WORKING_GAMMA), dtype=np.float32)


def working_oetf_decode(img: np.ndarray) -> np.ndarray:
    """Inverse of working_oetf_encode."""
    e = np.maximum(np.asarray(img, dtype=np.float32), 0.0)
    return np.power(e, np.float32(WORKING_GAMMA), dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance of (..., 3) data."""
    return LUMA_R * img[..., 0] + LUMA_G * img[..., 1] + LUMA_B * img[..., 2]


def adobe_to_srgb_linear(img: np.ndarray) -> np.ndarray:
    """Linear Adobe RGB primaries -> linear sRGB primaries (unclipped)."""
    flat = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    return (flat @ ADOBE_RGB_TO_SRGB.astype(np.float32).T).reshape(img.shape)


def encode_display(linear: np.ndarray, space: str = "sRGB") -> np.ndarray:
    """Encode a linear Adobe-primaries float image to float code values in [0,1].

    space: 'sRGB' (matrix + sRGB OETF), 'Adobe RGB' (x^(256/563)), 'Gray' (2D or 3-channel
    input reduced to luma, gamma 2.2).
    """
    if space == "sRGB":
        return linear_to_srgb(adobe_to_srgb_linear(linear))
    if space == "Adobe RGB":
        return working_oetf_encode(linear)
    if space == "Gray":
        y = linear if linear.ndim == 2 else luminance(np.asarray(linear, dtype=np.float32))
        return np.power(np.clip(y, 0.0, 1.0), np.float32(1.0 / GRAY_GAMMA), dtype=np.float32)
    raise ValueError(f"unknown output space {space!r}")


def to_uint(code: np.ndarray, bits: int = 16, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Quantize [0,1] float code values to uint16 (bits=16) or uint8 (bits=8)."""
    if bits == 16:
        dtype, scale = np.uint16, 65535.0
    elif bits == 8:
        dtype, scale = np.uint8, 255.0
    else:
        raise ValueError("bits must be 8 or 16")
    q = np.rint(np.clip(code, 0.0, 1.0) * np.float32(scale)).astype(dtype)
    if out is not None:
        out[...] = q
        return out
    return q


def encode_output(linear: np.ndarray, space: str = "sRGB", bits: int = 16, chunk_rows: Optional[int] = None) -> np.ndarray:
    """Linear Adobe-primaries image -> uint16/uint8 in the target space, processed in row chunks."""
    h = linear.shape[0]
    width = int(np.prod(linear.shape[1:])) or 1
    rows = chunk_rows or max(1, (1 << 20) // max(1, linear.shape[1] if linear.ndim > 1 else 1))
    out_shape = linear.shape[:2] if space == "Gray" else linear.shape
    out = np.empty(out_shape, dtype=np.uint16 if bits == 16 else np.uint8)
    for r in range(0, h, rows):
        to_uint(encode_display(linear[r : r + rows], space), bits, out=out[r : r + rows])
    return out
