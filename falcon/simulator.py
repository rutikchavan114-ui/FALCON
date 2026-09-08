from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ComponentTrajectory:
    component_id: str
    time: np.ndarray
    latent: np.ndarray
    full_observation: np.ndarray
    tau: float
    label: str


def rc_trajectory(time, tau=1.0, v0=5.0, vf=0.0):
    time = np.asarray(time, dtype=float)
    return vf + (v0 - vf) * np.exp(-time / tau)


def make_component(component_id, time, tau, noise_std, label, rng,
                   v0=5.0, vf=0.0):
    latent = rc_trajectory(time, tau=tau, v0=v0, vf=vf)
    observed = latent + rng.normal(0.0, noise_std, size=time.size)
    return ComponentTrajectory(
        component_id, np.asarray(time), latent, observed, tau, label
    )


def make_calibration_set(n_normal=30, time=None, rng=None):
    rng = np.random.default_rng(7) if rng is None else rng
    time = np.linspace(0.0, 5.0, 101) if time is None else np.asarray(time)
    out = []
    for i in range(n_normal):
        tau = max(0.85, rng.normal(1.0, 0.04))
        out.append(make_component(
            f"CAL-{i+1:02d}", time, tau, 0.015, "NORMAL", rng
        ))
    return out


def make_test_set(n_normal=4, n_anomaly=1, time=None, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    time = np.linspace(0.0, 5.0, 101) if time is None else np.asarray(time)
    out = []
    for i in range(n_normal):
        tau = max(0.85, rng.normal(1.0, 0.04))
        out.append(make_component(
            f"C{i+1:02d}", time, tau, 0.015, "NORMAL", rng
        ))
    for j in range(n_anomaly):
        out.append(make_component(
            f"C{n_normal+j+1:02d}", time, 0.47, 0.015, "ANOMALOUS", rng
        ))
    return out
