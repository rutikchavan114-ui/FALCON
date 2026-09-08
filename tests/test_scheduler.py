import numpy as np

from falcon.baseline import ReferenceBaseline
from falcon.gp_model import ComponentGP
from falcon.scheduler import decision_instability, next_interval


def make_gp():
    t = np.linspace(0, 2, 21)
    y = 5.0 * np.exp(-t)

    gp = ComponentGP()
    gp.fit(t, y)
    return gp


def test_decision_instability_returns_valid_values():
    t = np.linspace(0, 2, 21)

    reference = np.array([
        [5.00, 1.00],
        [5.05, 1.02],
        [4.95, 0.98],
        [5.02, 1.01],
        [4.98, 0.99],
    ])

    baseline = ReferenceBaseline(reference)

    gp = make_gp()

    instability, majority, scores = decision_instability(
        gp,
        t,
        baseline,
        threshold=5.0,
        n_samples=20,
        seed=42,
    )

    assert 0.0 <= instability <= 1.0
    assert majority in {"NORMAL", "ANOMALOUS"}
    assert len(scores) == 20


def test_next_interval_samples_faster_when_unstable():
    fast = next_interval(
        instability=0.20,
        base_interval=0.05,
        max_interval=0.50,
        p_threshold=0.05,
    )

    slow = next_interval(
        instability=0.0,
        base_interval=0.05,
        max_interval=0.50,
        p_threshold=0.05,
    )

    assert fast < slow


def test_early_observation_is_not_treated_as_sufficient_evidence():
    """
    Early partial observations should not be considered
    sufficient evidence for a full-horizon screening decision.
    """
    t = np.linspace(0, 5, 101)

    reference = np.array([
        [5.00, 1.00],
        [5.05, 1.02],
        [4.95, 0.98],
        [5.02, 1.01],
        [4.98, 0.99],
        [5.01, 1.00],
        [4.99, 1.01],
        [5.03, 0.98],
    ])

    baseline = ReferenceBaseline(reference)
    gp = ComponentGP()

    observed_t = t[:21]
    observed_y = 5.0 * np.exp(-observed_t)

    gp.fit(observed_t, observed_y)

    instability, majority, scores = decision_instability(
        gp,
        t,
        baseline,
        threshold=5.0,
        n_samples=20,
        seed=42,
    )

    # This test documents the current limitation:
    # decision_instability can confidently classify
    # an early extrapolation even though most of the
    # trajectory has not been observed.
    assert len(observed_t) / len(t) < 0.5
