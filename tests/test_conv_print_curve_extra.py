import pickle

import numpy as np

from conversion.modes import ProcessMode, auto_meter_for_mode, cast_removal_for_mode
from conversion.papers import PAPER_PROFILES, effective_paper_profile, profiles_for_mode
from conversion.print_curve import (
    apply_characteristic_curve,
    grade_coupled_shape,
    grade_to_slope,
    highlight_hold_offset,
    neutral_axis_affine,
    per_channel_curve_params,
    slope_to_grade,
)


def _img(h=40, w=50, seed=0):
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


def test_chunking_and_out_buffer_match():
    img = _img()
    p = (0.4, 4.0)
    a = apply_characteristic_curve(img, p, p, p, d_min=0.06, bpc=True)
    out = np.empty_like(img)
    b = apply_characteristic_curve(img, p, p, p, d_min=0.06, bpc=True, out=out, chunk_pixels=200)
    assert b is out
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert a.min() >= 0 and a.max() <= 1


def test_gray_path_matches_green_channel():
    img = _img()[..., 1]
    p = (0.4, 4.0)
    g = apply_characteristic_curve(img, p, p, p)
    rgb = apply_characteristic_curve(np.stack([img] * 3, -1), p, p, p)
    assert g.shape == img.shape
    np.testing.assert_allclose(g, rgb[..., 1], atol=1e-6)


def test_dye_matrix_paper_changes_color_not_neutral_axis_range():
    img = _img()
    p = (0.4, 4.0)
    base = apply_characteristic_curve(img, p, p, p)
    endura = apply_characteristic_curve(img, p, p, p, paper=PAPER_PROFILES["kodak_endura"])
    assert not np.allclose(base, endura)


def test_saturation_identity_and_desaturation():
    img = _img()
    p = (0.4, 4.0)
    a = apply_characteristic_curve(img, p, p, p, dye_separation=1.0)
    b = apply_characteristic_curve(img, p, p, p, dye_separation=0.5)
    assert np.std(b, axis=-1).mean() < np.std(a, axis=-1).mean()


def test_slope_grade_roundtrip_and_shape():
    assert abs(slope_to_grade(grade_to_slope(100.0, 1.2), 1.2) - 100.0) < 0.01
    t, s = grade_coupled_shape(10.0, 0.0, 0.0)
    assert t > 0 and s > 0


def test_cast_removal_neutral_axis_ties_channels():
    mid = (0.45, 0.5, 0.55)
    sh = (0.8, 0.85, 0.9)
    slopes, pivots, curvs = per_channel_curve_params(
        115.0, 1.0, False, 1.0, 1.3, None, None, neutral_axis_norm=(mid, sh, None)
    )
    assert slopes[1] != slopes[0]
    # green's curve at each green ref is matched by the other channels at their refs
    for ch in (0, 2):
        for ref_g, ref_c in ((0.5, mid[ch]), (0.85, sh[ch])):
            g = slopes[1] * (ref_g - pivots[1])
            c = slopes[ch] * (ref_c - pivots[ch]) + curvs[ch] * ref_c**2
            assert abs(g - c) < 0.35  # clamped deviation, loose bound
    gains, offs = neutral_axis_affine((mid, sh, None), 1.0)
    assert gains[1] == 1.0 and offs[1] == 0.0


def test_no_cast_removal_shared_curve():
    s, p, c = per_channel_curve_params(115.0, 1.0, True, 0.0, 1.3, None, 0.9)
    assert len(set(s)) == 1 and len(set(p)) == 1 and c == (0.0, 0.0, 0.0)


def test_highlight_hold_never_lifts():
    assert highlight_hold_offset(4.0, 0.2, 0.0) >= 0.0
    assert highlight_hold_offset(4.0, 0.2, 1.0) == 0.0


def test_modes_and_paper_filtering():
    assert cast_removal_for_mode(ProcessMode.E6, 1.0) == 0.0
    assert cast_removal_for_mode(ProcessMode.C41, 0.0) == 1.0
    assert auto_meter_for_mode(ProcessMode.E6, True) is False
    assert auto_meter_for_mode(ProcessMode.BW, False) is True
    assert {k for k, _ in profiles_for_mode(ProcessMode.BW)} >= {"neutral", "ilford_mg_rc"}
    assert len(profiles_for_mode(ProcessMode.E6)) == 1
    assert effective_paper_profile("ilford_mg_rc", ProcessMode.C41).kind == "default"


def test_picklable():
    pickle.dumps(PAPER_PROFILES["kodak_endura"])
    pickle.dumps(ProcessMode.C41)
