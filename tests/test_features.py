import numpy as np
from falcon.features import tau_63, normalize_trajectory


def test_rc_tau63():
    t = np.linspace(0, 5, 1001)
    y = 5 * np.exp(-t)
    assert abs(tau_63(t, y) - 1.0) < 0.02


def test_normalization():
    t = np.array([0.0, 1.0])
    y = np.array([5.0, 0.0])
    assert np.allclose(normalize_trajectory(t, y), [0.0, 1.0])
