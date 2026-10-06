"""Input decode: linear RAW via rawpy, non-RAW/Lightroom input linearization, display LUT.

Ported from NegPy (GPL-3.0): image_processor `_decode_sensor_rgb`, capture/raw_demosaic
`_user_sat`. RAW is decoded once with unity WB in camera primaries; as-shot white balance is
applied afterwards as a linear diagonal gain (`apply_wb`) so a film-type change never re-decodes.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from . import icc as icc_mod
from .output import srgb_to_linear

INPUT_ENCODINGS = ("Auto", "sRGB", "Linear")


# ------------------------------------------------------------------ RAW


def _linearity_limit(raw) -> int:
    """Camera's calibrated linear-range ceiling (raw ADC scale), falling back to white_level."""
    generic = int(raw.white_level or 0)
    per_channel = raw.camera_white_level_per_channel
    if not per_channel:
        return generic
    calibrated = int(min(per_channel))
    return min(calibrated, generic) if generic > 0 else calibrated


def _user_sat(raw) -> Optional[int]:
    """Linearity limit on the post-black-subtraction scale that postprocess() expects."""
    limit = _linearity_limit(raw)
    if limit <= 0:
        return None
    black = min(raw.black_level_per_channel or [0])
    return max(1, limit - black)


@dataclass
class RawMeta:
    camera_wb: Optional[Tuple[float, float, float]] = None  # R,G,B multipliers normalized to G
    rgb_xyz_matrix: Optional[np.ndarray] = None  # 3x3 XYZ->camera (libraw), None if absent
    colour_desc: Optional[str] = None


def raw_decode_kwargs(raw, params: dict, full_res: bool) -> dict:
    """rawpy.postprocess kwargs for the linear, unity-WB, camera-primaries decode.

    `params` holds dm_alg, fbdd_nr, median_filter_passes, noise_thr (RawProcessing.class_parameters).
    libraw's default flip is kept (saved rotation values depend on it).
    """
    import rawpy

    return dict(
        gamma=(1, 1),
        no_auto_bright=True,
        adjust_maximum_thr=0.0,
        user_sat=_user_sat(raw),
        output_color=rawpy.ColorSpace.raw,
        output_bps=16,
        use_camera_wb=False,
        user_wb=[1, 1, 1, 1],
        demosaic_algorithm=rawpy.DemosaicAlgorithm(params.get("dm_alg", 2)),
        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(params.get("fbdd_nr", 0)),
        median_filter_passes=params.get("median_filter_passes", 0),
        noise_thr=params.get("noise_thr", 0),
        half_size=not full_res,
    )


def extract_raw_meta(raw) -> RawMeta:
    wb = None
    try:
        w = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:3]
        if w.shape == (3,) and np.all(np.isfinite(w)) and w.min() > 0 and w[1] > 0:
            wb = tuple(float(v) for v in w / w[1])
    except Exception:
        pass
    mat = None
    try:
        m = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
        if m.ndim == 2 and m.shape[1] == 3 and np.any(m[:3] != 0):
            mat = m[:3].copy()
    except Exception:
        pass
    try:
        desc = raw.color_desc.decode("utf-8")
    except Exception:
        desc = None
    return RawMeta(wb, mat, desc)


def decode_raw(path: str, params: dict, full_res: bool = False) -> Tuple[np.ndarray, RawMeta]:
    """Decode a RAW to linear uint16 RGB (camera primaries, unity WB) plus metadata."""
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(**raw_decode_kwargs(raw, params, full_res))
        meta = extract_raw_meta(raw)
    return np.ascontiguousarray(rgb), meta


def apply_wb(lin: np.ndarray, wb: Optional[Sequence[float]], chunk_rows: int = 512) -> np.ndarray:
    """Multiply linear uint16 RGB by per-channel gains (normalized to green); returns uint16.

    A diagonal gain commutes with decode, so negatives can apply as-shot WB here after the
    unity-WB decode. Gains below 1 are normalized so the largest is 1 (no clipping).
    """
    if wb is None:
        return lin
    g = np.asarray(wb, dtype=np.float32)[:3]
    g = g / g.max()
    if np.allclose(g, 1.0):
        return lin
    out = np.empty_like(lin)
    for r in range(0, lin.shape[0], chunk_rows):
        out[r : r + chunk_rows] = np.clip(lin[r : r + chunk_rows] * g, 0, 65535).astype(np.uint16)
    return out


def black_border_box(display_gray: np.ndarray) -> Tuple[int, int, int, int]:
    """(x, y, w, h) of the non-black content of an 8-bit gray display copy (rawpy sometimes
    returns extra black border). Falls back to the full frame."""
    import cv2

    _, thresh = cv2.threshold(display_gray, 0, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, display_gray.shape[1], display_gray.shape[0]
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


# ------------------------------------------------------------------ display LUT

_DISPLAY_TARGET = 0.95
_DISPLAY_PCT = 99.5


def bt709_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099)


def display_gain(lin: np.ndarray, max_samples: int = 400_000) -> float:
    """Scalar gain putting the P99.5 of the frame's channel values at 0.95 (of 65535)."""
    flat = lin.reshape(-1)
    step = max(1, flat.shape[0] // max_samples)
    p = float(np.percentile(flat[::step], _DISPLAY_PCT)) / 65535.0
    return _DISPLAY_TARGET / p if p > 1e-6 else 1.0


def build_display_lut(gain: float = 1.0) -> np.ndarray:
    """65536-entry uint8 LUT: linear uint16 -> gain -> BT.709 OETF -> 8-bit. Monotonic."""
    x = np.arange(65536, dtype=np.float64) / 65535.0 * gain
    return np.rint(bt709_oetf(x) * 255.0).astype(np.uint8)


def apply_display_lut(lin_rgb: np.ndarray, lut: np.ndarray, bgr: bool = True, bits16: bool = False) -> np.ndarray:
    """Linear uint16 RGB -> display copy through the LUT (uint8 BGR by default)."""
    out = lut[lin_rgb]
    if bits16:
        out = out.astype(np.uint16) * 257
    if bgr and out.ndim == 3:
        out = np.ascontiguousarray(out[..., ::-1])
    return out


# ------------------------------------------------------------------ non-RAW input


@dataclass
class LoadedImage:
    lin: np.ndarray  # uint16 RGB, linear
    icc: Optional[bytes] = None  # source ICC profile
    in_adobe_primaries: bool = False  # True if converted to linear Adobe RGB primaries
    source_encoding: str = "sRGB"  # how it was linearized: 'icc', 'sRGB' or 'Linear'


def _read_pixels(path: str) -> Tuple[np.ndarray, Optional[bytes]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            tag = page.tags.get(34675)
            icc = bytes(tag.value) if tag is not None else None
        return arr, icc
    import cv2
    from PIL import Image

    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not isinstance(arr, np.ndarray):
        raise IOError(f"{path} failed to load")
    if arr.ndim == 3:
        arr = arr[..., :3][..., ::-1]  # BGR(A) -> RGB
    try:
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
    except Exception:
        icc = None
    return arr, icc


def load_non_raw(path: str, input_encoding: str = "Auto", to_adobe: bool = False) -> LoadedImage:
    """Read a TIFF/JPEG/PNG into linear uint16 RGB.

    With an embedded matrix/TRC ICC, its TRC linearizes the data (and `to_adobe` also rotates
    primaries to Adobe RGB, for slides). Without a usable ICC: 8-bit is treated as sRGB; 16-bit
    follows `input_encoding` (Auto = sRGB, sRGB, Linear). `to_adobe` for a profile-less file
    assumes sRGB primaries.
    """
    if input_encoding not in INPUT_ENCODINGS:
        raise ValueError(f"input_encoding must be one of {INPUT_ENCODINGS}")
    arr, icc = _read_pixels(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] < 3:
        raise ValueError("image must have at least three colour channels")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        f = arr.astype(np.float32) / 255.0
        is8 = True
    elif arr.dtype == np.uint16:
        f = arr.astype(np.float32) / 65535.0
        is8 = False
    else:
        f = np.clip(arr.astype(np.float32), 0.0, 1.0)
        is8 = False

    lin = icc_mod.linearize_with_icc(f, icc, to_adobe=to_adobe) if icc else None
    if lin is not None:
        enc, adobe = "icc", to_adobe
    else:
        use_srgb = is8 or input_encoding in ("Auto", "sRGB")
        lin = srgb_to_linear(f) if use_srgb else f
        enc, adobe = ("sRGB" if use_srgb else "Linear"), False
        if to_adobe:
            from .constants import ADOBE_RGB_TO_SRGB

            srgb_to_adobe = np.linalg.inv(ADOBE_RGB_TO_SRGB).astype(np.float32)
            lin = (lin.reshape(-1, 3) @ srgb_to_adobe.T).reshape(f.shape)
            adobe = True
    out = np.clip(np.rint(lin * 65535.0), 0, 65535).astype(np.uint16)
    return LoadedImage(np.ascontiguousarray(out), icc, adobe, enc)


def load_lightroom_tiff(path: str) -> LoadedImage:
    """Lightroom Edit-In output: 16-bit Adobe RGB with an embedded profile. Linearized via its
    TRC, primaries left as Adobe RGB. Export always embeds the bundled AdobeCompat profile."""
    return load_non_raw(path, input_encoding="Auto", to_adobe=True)


def lightroom_export_icc() -> bytes:
    return icc_mod.bundled_icc("Adobe RGB")
