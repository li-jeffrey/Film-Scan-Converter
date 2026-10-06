"""Minimal matrix/TRC ICC support: parse, bundle, and embed profiles.

Parses RGB matrix/TRC profiles (rXYZ/gXYZ/bXYZ + r/g/bTRC with 'curv' and 'para' curves) so
non-RAW input can be linearized, and injects/attaches ICC bytes to PNG, JPG and TIFF output.
"""
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_ASSET_DIR_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "icc"),
)

ICC_FILES = {
    "sRGB": "sRGB-v4.icc",
    "Adobe RGB": "AdobeCompat-v4.icc",
    "Gray": "GrayGamma2.2.icc",
}

_cache: dict = {}


def icc_dir() -> str:
    for d in _ASSET_DIR_CANDIDATES:
        d = os.path.normpath(d)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("bundled ICC directory not found")


def bundled_icc(space: str) -> bytes:
    """Bytes of the bundled profile for 'sRGB', 'Adobe RGB' or 'Gray'."""
    if space not in _cache:
        with open(os.path.join(icc_dir(), ICC_FILES[space]), "rb") as f:
            _cache[space] = f.read()
    return _cache[space]


# ------------------------------------------------------------------ parsing

# A TRC is ('gamma', g) | ('table', ndarray) | ('para', type, params) | ('identity',)


@dataclass(frozen=True)
class MatrixTrcProfile:
    primaries_xyz: np.ndarray  # 3x3, columns = r,g,b XYZ (PCS D50)
    trc: tuple  # three curves

    def linearize(self, code: np.ndarray) -> np.ndarray:
        """Per-channel TRC: encoded [0,1] -> linear."""
        out = np.empty(code.shape, dtype=np.float32)
        for ch in range(3):
            out[..., ch] = eval_trc(self.trc[ch], code[..., ch])
        return out


def _tags(data: bytes) -> dict:
    n = struct.unpack(">I", data[128:132])[0]
    tags = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", data[132 + 12 * i : 144 + 12 * i])
        tags[sig.decode("latin1")] = data[off : off + size]
    return tags


def _s15(b: bytes) -> float:
    return struct.unpack(">i", b)[0] / 65536.0


def _parse_xyz(tag: bytes) -> np.ndarray:
    return np.array([_s15(tag[8 + 4 * i : 12 + 4 * i]) for i in range(3)], dtype=np.float64)


def _parse_curve(tag: bytes) -> tuple:
    sig = tag[:4]
    if sig == b"curv":
        n = struct.unpack(">I", tag[8:12])[0]
        if n == 0:
            return ("identity",)
        if n == 1:
            return ("gamma", struct.unpack(">H", tag[12:14])[0] / 256.0)
        vals = np.frombuffer(tag[12 : 12 + 2 * n], dtype=">u2").astype(np.float64) / 65535.0
        return ("table", vals)
    if sig == b"para":
        ftype = struct.unpack(">H", tag[8:10])[0]
        count = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ftype]
        params = tuple(_s15(tag[12 + 4 * i : 16 + 4 * i]) for i in range(count))
        return ("para", ftype, params)
    raise ValueError(f"unsupported curve type {sig!r}")


def parse_icc(data: bytes) -> Optional[MatrixTrcProfile]:
    """Parse a matrix/TRC RGB profile; None if the profile is not of that kind."""
    try:
        if data[16:20] != b"RGB ":
            return None
        t = _tags(data)
        cols = [_parse_xyz(t[k]) for k in ("rXYZ", "gXYZ", "bXYZ")]
        trc = tuple(_parse_curve(t[k]) for k in ("rTRC", "gTRC", "bTRC"))
    except (KeyError, struct.error, ValueError, IndexError):
        return None
    return MatrixTrcProfile(np.stack(cols, axis=1), trc)


def eval_trc(curve: tuple, x: np.ndarray) -> np.ndarray:
    """Evaluate a parsed curve: encoded -> linear."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    kind = curve[0]
    if kind == "identity":
        return x
    if kind == "gamma":
        return np.power(x, np.float32(curve[1]), dtype=np.float32)
    if kind == "table":
        tbl = curve[1]
        return np.interp(x, np.linspace(0.0, 1.0, len(tbl)), tbl).astype(np.float32)
    _, ftype, p = curve
    g = p[0]
    xd = x.astype(np.float64)
    if ftype == 0:
        y = xd**g
    else:
        a, b = p[1], p[2]
        if ftype == 1:
            y = np.where(xd >= -b / a, (a * xd + b).clip(0) ** g, 0.0)
        elif ftype == 2:
            c = p[3]
            y = np.where(xd >= -b / a, (a * xd + b).clip(0) ** g + c, c)
        elif ftype == 3:
            c, d = p[3], p[4]
            y = np.where(xd >= d, (a * xd + b).clip(0) ** g, c * xd)
        else:
            c, d, e, f = p[3], p[4], p[5], p[6]
            y = np.where(xd >= d, (a * xd + b).clip(0) ** g + e, c * xd + f)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


# D50 -> D65 Bradford, for primaries given relative to the ICC D50 PCS
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
_D50 = np.array([0.9642, 1.0, 0.8249])
_D65 = np.array([0.95047, 1.0, 1.08883])


def bradford_adapt(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    s, d = _BRADFORD @ src_white, _BRADFORD @ dst_white
    return np.linalg.inv(_BRADFORD) @ np.diag(d / s) @ _BRADFORD


def profile_to_linear_adobe_matrix(profile: MatrixTrcProfile) -> np.ndarray:
    """3x3 taking linear profile RGB -> linear Adobe RGB (D65) primaries."""
    from .constants import XYZ_TO_ADOBE_RGB

    to_d65 = bradford_adapt(_D50, _D65) @ profile.primaries_xyz
    return XYZ_TO_ADOBE_RGB @ to_d65


def linearize_with_icc(rgb: np.ndarray, icc: Optional[bytes], to_adobe: bool = False) -> Optional[np.ndarray]:
    """Apply an embedded profile to float [0,1] RGB. Returns linear float32 (in Adobe primaries
    if `to_adobe`), or None if the profile cannot be parsed."""
    prof = parse_icc(icc) if icc else None
    if prof is None:
        return None
    lin = prof.linearize(rgb)
    if to_adobe:
        m = profile_to_linear_adobe_matrix(prof).astype(np.float32)
        lin = (lin.reshape(-1, 3) @ m.T).reshape(lin.shape)
    return lin


# ------------------------------------------------------------------ embedding

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def inject_png_iccp(png: bytes, icc: bytes, name: str = "ICC") -> bytes:
    """Insert (or replace) an iCCP chunk right after IHDR."""
    if not png.startswith(_PNG_SIG):
        raise ValueError("not a PNG")
    out, pos, inserted = [_PNG_SIG], len(_PNG_SIG), False
    while pos < len(png):
        n = struct.unpack(">I", png[pos : pos + 4])[0]
        kind = png[pos + 4 : pos + 8]
        raw = png[pos : pos + 12 + n]
        pos += 12 + n
        if kind in (b"iCCP", b"sRGB"):
            continue
        out.append(raw)
        if kind == b"IHDR" and not inserted:
            out.append(_chunk(b"iCCP", name.encode("latin1") + b"\x00\x00" + zlib.compress(icc)))
            inserted = True
    return b"".join(out)


def extract_png_iccp(png: bytes) -> Optional[bytes]:
    pos = len(_PNG_SIG)
    while pos < len(png):
        n = struct.unpack(">I", png[pos : pos + 4])[0]
        kind = png[pos + 4 : pos + 8]
        if kind == b"iCCP":
            body = png[pos + 8 : pos + 8 + n]
            i = body.index(b"\x00")
            return zlib.decompress(body[i + 2 :])
        pos += 12 + n
    return None


def write_image(path: str, img: np.ndarray, filetype: str, icc: Optional[bytes], *, jpg_quality: int = 90, tiff_compression: int = 8) -> None:
    """Write an RGB (not BGR) uint8/uint16 image or 2-D gray with an embedded ICC profile.

    JPG: PIL (8-bit; 16-bit input is reduced). TIFF: tifffile. PNG: cv2 then iCCP injection.
    """
    ft = filetype.upper()
    if ft in ("JPG", "JPEG"):
        from PIL import Image

        a = img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8)
        Image.fromarray(a).save(path, "JPEG", quality=int(jpg_quality), icc_profile=icc)
    elif ft in ("TIF", "TIFF"):
        import tifffile

        comp = {1: None, 5: "lzw", 8: "deflate", 32946: "deflate"}.get(int(tiff_compression), "deflate")
        tifffile.imwrite(path, img, photometric="minisblack" if img.ndim == 2 else "rgb", compression=comp, iccprofile=icc)
    elif ft == "PNG":
        import cv2

        bgr = img if img.ndim == 2 else np.ascontiguousarray(img[..., ::-1])
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise IOError("PNG encode failed")
        data = buf.tobytes()
        if icc:
            data = inject_png_iccp(data, icc)
        with open(path, "wb") as f:
            f.write(data)
    else:
        raise ValueError(f"unsupported filetype {filetype!r}")


def tiff_icc(path: str) -> Optional[bytes]:
    """ICC profile bytes (tag 34675) from a TIFF, or None."""
    import tifffile

    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags.get(34675)
        return bytes(tag.value) if tag is not None else None
