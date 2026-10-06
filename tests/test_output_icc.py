"""working OETF (ported from NegPy test_working_oetf), output encoding, ICC parse/embed."""
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from conversion import icc
from conversion.constants import WORKING_GAMMA
from conversion.output import (
    adobe_to_srgb_linear,
    encode_display,
    encode_output,
    linear_to_srgb,
    srgb_to_linear,
    working_oetf_decode,
    working_oetf_encode,
)


def test_oetf_known_values():
    x = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(working_oetf_encode(x)[0, 0], [0, 0.5 ** (1 / WORKING_GAMMA), 1], atol=1e-5)


def test_oetf_roundtrip_and_clamp():
    ramp = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, -1, 1) * np.ones((1, 1, 3), np.float32)
    np.testing.assert_allclose(working_oetf_decode(working_oetf_encode(ramp)), ramp, atol=1e-5)
    e = working_oetf_encode(np.array([[[-0.5, 1.5, 0.2]]], np.float32))
    assert e.min() >= 0 and e.max() <= 1 and e.dtype == np.float32


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)


def test_adobe_to_srgb_neutral_preserved():
    g = np.full((2, 2, 3), 0.4, np.float32)
    np.testing.assert_allclose(adobe_to_srgb_linear(g), g, atol=1e-4)


def test_encode_output_dtypes_and_gray():
    img = np.random.default_rng(0).random((17, 9, 3)).astype(np.float32)
    assert encode_output(img, "sRGB", 16).dtype == np.uint16
    assert encode_output(img, "Adobe RGB", 8, chunk_rows=4).dtype == np.uint8
    g = encode_output(img, "Gray", 16)
    assert g.shape == (17, 9)
    # chunked == unchunked
    np.testing.assert_array_equal(encode_output(img, "sRGB", 16, chunk_rows=3), encode_output(img, "sRGB", 16, chunk_rows=100))


def test_bundled_profiles_parse():
    for space in ("sRGB", "Adobe RGB"):
        p = icc.parse_icc(icc.bundled_icc(space))
        assert p is not None
    assert icc.parse_icc(icc.bundled_icc("Gray")) is None  # gray profile, not RGB matrix/TRC


def test_parsed_srgb_trc_matches_srgb_oetf():
    p = icc.parse_icc(icc.bundled_icc("sRGB"))
    x = np.linspace(0, 1, 33, dtype=np.float32)
    rgb = np.stack([x, x, x], -1)
    np.testing.assert_allclose(p.linearize(rgb)[:, 1], srgb_to_linear(x), atol=2e-3)


def test_parsed_adobe_trc_and_matrix():
    p = icc.parse_icc(icc.bundled_icc("Adobe RGB"))
    x = np.linspace(0, 1, 33, dtype=np.float32)
    rgb = np.stack([x, x, x], -1)
    np.testing.assert_allclose(p.linearize(rgb)[:, 0], x**WORKING_GAMMA, atol=2e-3)
    m = icc.profile_to_linear_adobe_matrix(p)
    np.testing.assert_allclose(m, np.eye(3), atol=2e-3)  # Adobe -> Adobe is identity


def test_srgb_profile_to_adobe_matches_constants():
    from conversion.constants import ADOBE_RGB_TO_SRGB

    p = icc.parse_icc(icc.bundled_icc("sRGB"))
    m = icc.profile_to_linear_adobe_matrix(p)
    np.testing.assert_allclose(m, np.linalg.inv(ADOBE_RGB_TO_SRGB), atol=3e-3)


def test_para_curve_types():
    # type 3 sRGB-like: g=2.4 a=1/1.055 b=.055/1.055 c=1/12.92 d=.04045
    c = ("para", 3, (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045))
    x = np.linspace(0, 1, 11, dtype=np.float32)
    np.testing.assert_allclose(icc.eval_trc(c, x), srgb_to_linear(x), atol=1e-5)
    np.testing.assert_allclose(icc.eval_trc(("para", 0, (2.0,)), x), x**2, atol=1e-6)


def test_png_iccp_roundtrip(tmp_path):
    data = icc.bundled_icc("sRGB")
    img = (np.arange(4 * 5 * 3).reshape(4, 5, 3) * 1000).astype(np.uint16)
    p = tmp_path / "a.png"
    icc.write_image(str(p), img, "PNG", data)
    raw = p.read_bytes()
    assert icc.extract_png_iccp(raw) == data
    import cv2

    back = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., ::-1]
    assert back.dtype == np.uint16 and back.shape == img.shape
    np.testing.assert_array_equal(back, img)  # RGB order preserved


def test_tiff_and_jpg_embed(tmp_path):
    data = icc.bundled_icc("Adobe RGB")
    img = np.random.default_rng(1).integers(0, 65535, (8, 8, 3), dtype=np.uint16)
    t = tmp_path / "a.tif"
    icc.write_image(str(t), img, "TIFF", data)
    assert icc.tiff_icc(str(t)) == data
    np.testing.assert_array_equal(tifffile.imread(t), img)
    j = tmp_path / "a.jpg"
    icc.write_image(str(j), img, "JPG", data, jpg_quality=90)
    assert Image.open(j).info["icc_profile"] == data
    g = tmp_path / "g.tif"
    icc.write_image(str(g), img[..., 0], "TIFF", icc.bundled_icc("Gray"))
    assert icc.tiff_icc(str(g)) == icc.bundled_icc("Gray")


def test_encoded_srgb_matches_littlecms():
    """Our sRGB encode of Adobe-primaries data ~ an lcms Adobe->sRGB transform."""
    rng = np.random.default_rng(2)
    lin = rng.random((16, 16, 3)).astype(np.float32) * 0.6 + 0.2
    ours = (encode_display(lin, "sRGB") * 255).round().astype(np.uint8)
    adobe8 = (working_oetf_encode(lin) * 255).round().astype(np.uint8)
    xf = ImageCms.buildTransform(
        ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc.bundled_icc("Adobe RGB"))),
        ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc.bundled_icc("sRGB"))),
        "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    ref = np.asarray(ImageCms.applyTransform(Image.fromarray(adobe8), xf))
    assert np.abs(ours.astype(int) - ref.astype(int)).mean() < 2.0
