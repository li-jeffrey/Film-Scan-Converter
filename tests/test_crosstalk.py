import pickle

import numpy as np

from conversion import crosstalk as ct
from conversion.density import unmix_log_image


def test_bundled_profiles_load():
    names = ct.list_profiles()
    assert names[0] == "Generic C41"
    assert len(names) == 17  # generic + 16 bundled TOMLs
    m = ct.get_matrix("Kodak Portra 400 (approx)")
    assert m is not None and len(m) == 9 and m[0] == 1.0


def test_default_returns_none_and_unknown_none():
    assert ct.get_matrix("Generic C41") is None
    assert ct.get_matrix("nope") is None


def test_missing_process_key_is_c41():
    for name in ct.list_profiles():
        assert ct.get_process(name) == str(ct.ProcessMode.C41)


def test_resolve_off_and_row_normalised():
    assert ct.resolve_crosstalk_matrix(0.0, None) is None
    m = ct.resolve_crosstalk_matrix(1.0, None)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    m0 = ct.resolve_crosstalk_matrix(1e-9, None)
    np.testing.assert_allclose(m0, np.eye(3), atol=1e-6)


def test_process_gate():
    assert ct.effective_crosstalk_matrix(1.0, None, None, str(ct.ProcessMode.E6)) is None
    assert ct.effective_crosstalk_matrix(1.0, None, None, str(ct.ProcessMode.C41)) is not None
    assert ct.effective_crosstalk_matrix(1.0, None, ct.ProcessMode.E6, ct.ProcessMode.E6) is not None
    assert ct.list_profiles(ct.ProcessMode.E6) == []


def test_legacy_mode_names():
    assert ct.ProcessMode("E-6") == ct.ProcessMode.E6
    assert ct.ProcessMode("garbage") == ct.ProcessMode.C41


def test_unmix_preserves_gray_and_matches_matmul():
    m = ct.resolve_crosstalk_matrix(0.7, None)
    gray = np.full((4, 5, 3), -0.8, np.float32)
    np.testing.assert_allclose(unmix_log_image(gray, m), gray, atol=1e-5)
    rng = np.random.default_rng(0)
    img = rng.normal(size=(6, 7, 3)).astype(np.float32)
    np.testing.assert_allclose(unmix_log_image(img, m), np.einsum("hwc,kc->hwk", img, m), atol=1e-5)
    assert unmix_log_image(img, None) is img


def test_picklable_matrix():
    m = ct.resolve_crosstalk_matrix(0.5, None)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(m)), m)
