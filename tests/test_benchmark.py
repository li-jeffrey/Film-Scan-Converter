import time

import numpy as np
import pytest

from conversion import pipeline
from conversion.params import ConversionParams


@pytest.mark.slow
def test_render_24mp_benchmark():
    rng = np.random.default_rng(0)
    h, w = 4000, 6000
    ramp = np.linspace(0.05, 0.6, w, dtype=np.float32)[None, :, None]
    lin = np.empty((h, w, 3), np.uint16)
    for c, k in enumerate((0.9, 0.7, 0.5)):
        lin[..., c] = (np.clip(ramp[..., 0] * k + rng.normal(0, 0.002, (1, w)), 0.01, 1) * 65535).astype(np.uint16)
    small = lin[::8, ::8]
    p = ConversionParams()
    a = pipeline.analyze(pipeline.build_grid(small, p), p)
    t = time.perf_counter()
    out = pipeline.render(lin, p, a, bgr=True)
    dt = time.perf_counter() - t
    print(f'24 MP render: {dt:.1f}s')
    assert out.shape == (h, w, 3) and dt < 60
