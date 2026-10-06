import numpy as np

_G = 256.0 / 563.0


def working_oetf_encode(x):
    return np.power(np.clip(np.asarray(x, dtype=np.float32), 0.0, None), _G)


def working_oetf_decode(x):
    return np.power(np.clip(np.asarray(x, dtype=np.float32), 0.0, None), 1.0 / _G)
