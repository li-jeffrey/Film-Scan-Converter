import dataclasses

import numpy as np
import pytest

from conversion import pipeline
from conversion.params import ConversionParams, FILM_TYPE_BW, FILM_TYPE_COLOUR, FILM_TYPE_SLIDE

BASE = np.array([0.25, 0.6, 0.9])  # orange-mask density


def synth_negative(h=240, w=320, seed=0):
    """Gray ramp + colour patches photographed on an orange-masked negative (linear 0..1)."""
    rng = np.random.default_rng(seed)
    scene = np.tile(np.linspace(0.02, 0.9, w, dtype=np.float32)[None, :, None], (h, 1, 3))
    scene[: h // 4, : w // 4] = (0.7, 0.2, 0.2)
    scene[: h // 4, w // 4: w // 2] = (0.2, 0.6, 0.2)
    scene[: h // 4, w // 2: 3 * w // 4] = (0.2, 0.2, 0.7)
    scene = np.clip(scene + rng.normal(0, 0.003, scene.shape), 1e-3, 1).astype(np.float32)
    dens = 0.6 * (np.log10(scene) + 3.0) + BASE  # density rises with exposure
    return (10.0 ** -dens).astype(np.float32), scene


def run(params, lin, **kw):
    grid = pipeline.build_grid(lin, params)
    a = pipeline.analyze(grid, params)
    return pipeline.render(lin, params, a, **kw), a


def test_colour_negative_smoke():
    lin, scene = synth_negative()
    out, a = run(ConversionParams(), lin)
    assert out.dtype == np.uint16 and out.shape == lin.shape
    assert np.isfinite(out).all()
    ramp = out[h_ramp(lin)].astype(np.float64) / 65535.0
    g = ramp.mean(axis=0)  # (w, 3)
    lum = g.mean(axis=1)
    # positive: scene ramp goes dark->bright left to right
    assert lum[-10:].mean() > lum[:10].mean() + 0.2
    assert np.all(np.diff(np.convolve(lum, np.ones(15) / 15, "valid")) > -0.01)
    chroma = (g.max(axis=1) - g.min(axis=1))[20:-20]
    assert chroma.mean() < 0.05


def h_ramp(lin):
    h, w = lin.shape[:2]
    return slice(h // 2, h), slice(0, w)


def test_bw_smoke_gray_2d():
    lin, _ = synth_negative()
    out, _ = run(ConversionParams(film_type=FILM_TYPE_BW, paper_profile="neutral"), lin)
    assert out.ndim == 2 and out.dtype == np.uint16 and np.isfinite(out).all()


def test_chunked_equals_full_and_bgr():
    lin, _ = synth_negative()
    p = ConversionParams()
    grid = pipeline.build_grid(lin, p)
    a = pipeline.analyze(grid, p)
    a1 = pipeline.render(lin, p, a, chunk_pixels=1 << 30)
    a2 = pipeline.render(lin, p, a, chunk_pixels=5000)
    assert np.array_equal(a1, a2)
    b = pipeline.render(lin, p, a, bgr=True)
    assert np.array_equal(b, a1[..., ::-1])
    n = pipeline.normalize(lin, p, a)
    assert np.array_equal(pipeline.render(None, p, a, norm=n), a1)


def test_hue_order_preserved_patches():
    lin, scene = synth_negative()
    out, _ = run(ConversionParams(), lin)
    h, w = lin.shape[:2]
    red = out[h // 8, w // 8].astype(float)
    grn = out[h // 8, 3 * w // 8].astype(float)
    blu = out[h // 8, 5 * w // 8].astype(float)
    assert red.argmax() == 0 and grn.argmax() == 1 and blu.argmax() == 2


def test_slide_runs():
    lin, _ = synth_negative()
    pos = (10.0 ** -(0.6 * (-np.log10(lin)))).astype(np.float32)  # arbitrary positive-ish
    p = ConversionParams(film_type=FILM_TYPE_SLIDE, auto_density=False, auto_grade=False, cast_removal=0.0)
    out, _ = run(p, pos)
    assert out.shape == pos.shape and np.isfinite(out).all()


def test_effective_grade_and_params_keys():
    lin, _ = synth_negative()
    p = ConversionParams()
    a = pipeline.analyze(pipeline.build_grid(lin, p), p)
    g = pipeline.effective_grade(p, a)
    assert g is not None and 50 <= g <= 180


def test_base_override_changes_bounds():
    lin, _ = synth_negative()
    p = ConversionParams()
    a0 = pipeline.analyze(pipeline.build_grid(lin, p), p)
    p2 = dataclasses.replace(p, base_rgb=(0.4, 0.2, 0.1))
    a1 = pipeline.analyze(pipeline.build_grid(lin, p2), p2)
    assert a0.bounds.ceils != a1.bounds.ceils


def test_analysis_picklable_small():
    import pickle
    lin, _ = synth_negative()
    p = ConversionParams()
    a = pipeline.analyze(pipeline.build_grid(lin, p), p)
    assert len(pickle.dumps(a)) < 5000
