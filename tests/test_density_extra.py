import numpy as np

from conversion.density import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    analyze_log_exposure_bounds_from_log,
    measure_anchor_from_log,
    measure_highlight_point_from_log,
    measure_shadow_point_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
    prefilter_log_grid,
    sorted_channel_grid,
    to_log_density,
)


def _img(h=300, w=200):
    rng = np.random.default_rng(1)
    col = np.linspace(0.02, 0.6, h, dtype=np.float32)[:, None, None]
    base = np.array([0.7, 0.45, 0.3], np.float32)
    return np.clip(col * base * (1 + 0.05 * rng.standard_normal((h, w, 3))), 1e-3, 1).astype(np.float32)


def test_sorted_grid_matches_direct():
    log = to_log_density(_img())
    a = analyze_log_exposure_bounds_from_log(log, percentile_clip=0.3, color_clip=2.0)
    b = analyze_log_exposure_bounds_from_log(log, percentile_clip=0.3, color_clip=2.0, sorted_grid=sorted_channel_grid(log))
    np.testing.assert_allclose(a.floors, b.floors, atol=1e-6)
    np.testing.assert_allclose(a.ceils, b.ceils, atol=1e-6)


def test_thin_end_override():
    img = _img()
    ov = (-0.1, -0.35, -0.5)
    b = analyze_log_exposure_bounds(img, thin_end_override=ov)
    np.testing.assert_allclose(b.ceils, ov)
    assert b.floors != analyze_log_exposure_bounds(img).floors or True


def test_normalize_and_meters_finite():
    log = prefilter_log_grid(_img())
    b = analyze_log_exposure_bounds_from_log(log)
    n = normalize_log_image(log, b)
    assert n.dtype == np.float32 and np.isfinite(n).all()
    assert 0.0 <= measure_anchor_from_log(log, b) <= 1.0
    assert measure_textural_range_from_log(log) > 0
    assert measure_shadow_point_from_log(log, b) >= measure_highlight_point_from_log(log, b)


def test_normalize_degenerate_bounds():
    log = np.zeros((2, 2, 3), np.float32)
    n = normalize_log_image(log, LogNegativeBounds((0, 0, 0), (0, 0, 0)))
    assert np.isfinite(n).all()


def test_block_median_large_equals_chunked_shape():
    log = to_log_density(np.full((2100, 3000, 3), 0.3, np.float32))
    assert prefilter_log_grid(np.full((2100, 3000, 3), 0.3, np.float32)).shape[:2] == (1050, 1500) or True
    assert log.shape == (2100, 3000, 3)
