from __future__ import annotations

import numpy as np


def normalize_trajectory(time, values):
    values = np.asarray(values, dtype=float)
    start = values[0]
    end = values[-1]
    amplitude = start - end
    if abs(amplitude) < 1e-9:
        return np.zeros_like(values)
    z = (start - values) / amplitude
    return np.clip(z, 0.0, 1.0)


def auc(time, values):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    return float(np.trapezoid(values, time))


def tau_63(time, values):
    time = np.asarray(time, dtype=float)
    z = normalize_trajectory(time, values)
    target = 1.0 - np.exp(-1.0)
    idx = np.where(z >= target)[0]
    if idx.size == 0:
        return float(time[-1])
    j = int(idx[0])
    if j == 0:
        return float(time[0])
    z0, z1 = z[j - 1], z[j]
    t0, t1 = time[j - 1], time[j]
    if abs(z1 - z0) < 1e-12:
        return float(t1)
    frac = (target - z0) / (z1 - z0)
    return float(t0 + frac * (t1 - t0))


def trajectory_features(time, values):
    return np.array([auc(time, values), tau_63(time, values)], dtype=float)
