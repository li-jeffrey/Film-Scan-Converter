import numpy as np
import pytest
import tifffile
from PIL import Image

from conversion import decode, icc
from conversion.constants import TRANSFER_DENSITY_RANGE
from conversion.transfer import (
    apply_transfer_curve,
    display_rendering,
    transfer_bounds,
    transfer_curve_params,
    zone_geometry,
)


class Cfg:
    density = 1.0
    grade = 115.0
    toe = 0.0
    shoulder = 0.0
    toe_width = 2.5
    shoulder_width = 2.5


# ---- transfer
def test_transfer_identity_positive_source():
    rng = np.random.default_rng(0)
    scene = rng.uniform(0.02, 0.9, (4, 4, 3)).astype(np.float32)
    n = -np.log10(scene) / TRANSFER_DENSITY_RANGE
    off, contrast, toe, sh = transfer_curve_params(Cfg)
    assert off == 0.0 and contrast == 1.0
    out = apply_transfer_curve(n.astype(np.float32), off, contrast, toe, sh, positive_source=True)
    np.testing.assert_allclose(out, scene, rtol=1e-5)


def test_transfer_display_rendering_monotonic_bounded():
    x = np.linspace(0, 20, 500, dtype=np.float32)
    y = display_rendering(x)
    assert y[0] == 0 and y.max() <= 1 and np.all(np.diff(y) >= -1e-7)


def test_transfer_density_darkens_and_knees_monotonic():
    n = np.linspace(0.05, 0.95, 64, dtype=np.float32).reshape(1, -1, 1) * np.ones((1, 1, 3), np.float32)
    base = apply_transfer_curve(n, 0.0, 1.0, (0,) * 3, (0,) * 3)
    dark = apply_transfer_curve(n, -0.3, 1.0, (0,) * 3, (0,) * 3)
    assert dark.mean() < base.mean()
    shaped = apply_transfer_curve(n, 0.0, 1.0, (0.5,) * 3, (0.5,) * 3, shadow_density=0.3, highlight_density=-0.2)
    assert np.all(np.diff(shaped[0, :, 1]) <= 1e-6)  # brighter input density -> darker output


def test_transfer_separation_and_damping_run():
    n = np.random.default_rng(1).random((5, 5, 3)).astype(np.float32)
    a = apply_transfer_curve(n, 0, 1.0, (0,) * 3, (0,) * 3, separation=1.2, damping=0.5)
    assert a.shape == n.shape and np.isfinite(a).all()


def test_transfer_bounds_and_zone():
    f, c = transfer_bounds()
    assert f == (0, 0, 0) and c == (-3.0,) * 3
    s, h, k = zone_geometry()
    assert s > h > 0 and k > 0


# ---- display LUT
def test_display_lut_monotonic_and_gain():
    lut = decode.build_display_lut(1.0)
    assert lut.shape == (65536,) and lut[0] == 0 and lut[-1] == 255 and np.all(np.diff(lut.astype(int)) >= 0)
    lin = np.random.default_rng(0).integers(0, 20000, (64, 64, 3), dtype=np.uint16)
    g = decode.display_gain(lin)
    assert g > 1
    disp = decode.apply_display_lut(lin, decode.build_display_lut(g))
    assert disp.dtype == np.uint8 and disp.shape == lin.shape


def test_apply_wb_diagonal():
    lin = np.full((4, 4, 3), 10000, np.uint16)
    out = decode.apply_wb(lin, (2.0, 1.0, 1.5))
    assert out[0, 0].tolist() == [10000, 5000, 7500]
    assert decode.apply_wb(lin, None) is lin


# ---- non-RAW load
def test_load_8bit_jpg_srgb(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(p)
    li = decode.load_non_raw(str(p))
    assert li.source_encoding == "sRGB" and li.lin.dtype == np.uint16
    assert abs(li.lin[0, 0, 0] / 65535 - 0.2158) < 1e-3


def test_load_16bit_no_icc_encodings(tmp_path):
    p = tmp_path / "a.tif"
    tifffile.imwrite(p, np.full((4, 4, 3), 32768, np.uint16), photometric="rgb")
    assert decode.load_non_raw(str(p), "Linear").lin[0, 0, 0] == 32768
    assert abs(decode.load_non_raw(str(p), "sRGB").lin[0, 0, 0] / 65535 - 0.2140) < 1e-3
    assert decode.load_non_raw(str(p), "Auto").source_encoding == "sRGB"


def test_load_tiff_with_adobe_icc_and_lightroom(tmp_path):
    p = tmp_path / "lr.tif"
    code = np.full((4, 4, 3), 32768, np.uint16)
    icc.write_image(str(p), code, "TIFF", icc.bundled_icc("Adobe RGB"))
    li = decode.load_lightroom_tiff(str(p))
    assert li.source_encoding == "icc" and li.in_adobe_primaries
    assert abs(li.lin[0, 0, 1] / 65535 - 0.5 ** 2.19921875) < 2e-3
    assert li.icc == icc.bundled_icc("Adobe RGB")
    assert decode.lightroom_export_icc() == icc.bundled_icc("Adobe RGB")


def test_srgb_icc_to_adobe_slide_gamut(tmp_path):
    p = tmp_path / "s.tif"
    code = np.zeros((2, 2, 3), np.uint16)
    code[..., 0] = 65535  # pure sRGB red
    icc.write_image(str(p), code, "TIFF", icc.bundled_icc("sRGB"))
    li = decode.load_non_raw(str(p), to_adobe=True)
    r = li.lin[0, 0] / 65535
    # sRGB red in linear Adobe RGB is ~(0.715, 0, 0)
    np.testing.assert_allclose(r, [0.715, 0.0, 0.0], atol=0.01)


def test_raw_kwargs_shape():
    class FakeRaw:
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]
        black_level_per_channel = [512] * 4

    kw = decode.raw_decode_kwargs(FakeRaw(), {}, full_res=False)
    assert kw["gamma"] == (1, 1) and kw["no_auto_bright"] and kw["adjust_maximum_thr"] == 0.0
    assert kw["user_sat"] == 15311 - 512 and kw["user_wb"] == [1, 1, 1, 1] and kw["half_size"]
    assert kw["output_bps"] == 16 and kw["use_camera_wb"] is False
    assert decode.raw_decode_kwargs(FakeRaw(), {}, True)["half_size"] is False
