from __future__ import annotations

import numpy as np

from .features import trajectory_features
from .baseline import ReferenceBaseline
from .anomaly import classify


def decision_instability(gp, t_grid, baseline, threshold,
                         n_samples=50, seed=0):
    samples = gp.sample_trajectories(
        t_grid, n_samples=n_samples, rng_seed=seed
    )
    decisions = []
    scores = []

    for trajectory in samples:
        f = trajectory_features(t_grid, trajectory)
        score = baseline.score(f)
        scores.append(score)
        decisions.append(classify(score, threshold))

    decisions = np.asarray(decisions)
    scores = np.asarray(scores)

    normal_count = np.sum(decisions == "NORMAL")
    anomalous_count = np.sum(decisions == "ANOMALOUS")
    majority = "ANOMALOUS" if anomalous_count > normal_count else "NORMAL"
    instability = float(np.mean(decisions != majority))
    return instability, majority, scores


def next_interval(instability, base_interval, max_interval,
                  p_threshold=0.05, growth=2.0):
    if instability > p_threshold:
        return base_interval
    return min(max_interval, base_interval * growth)


def evidence_sufficient(
    observed_time,
    full_time,
    minimum_fraction=0.50,
):
    """
    Return True only when a sufficient fraction of the
    total observation window has actually been observed.

    This prevents early GP extrapolation from being treated
    as sufficient evidence for final trajectory classification.
    """
    observed_time = float(observed_time)
    full_time = np.asarray(full_time, dtype=float)

    if full_time.size < 2:
        return True

    start = float(full_time[0])
    end = float(full_time[-1])

    duration = end - start

    if duration <= 0:
        return True

    observed_fraction = (
        (observed_time - start) / duration
    )

    return observed_fraction >= minimum_fraction
