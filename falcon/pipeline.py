
import numpy as np

from .gp_model import ComponentGP
from .features import trajectory_features
from .scheduler import (
    decision_instability,
    next_interval,
    evidence_sufficient,
)
from .baseline import ReferenceBaseline
from .anomaly import classify


def build_reference(calibration_components, dense_grid, gp_kwargs=None):
    gp_kwargs = {} if gp_kwargs is None else gp_kwargs

    rows = []

    for comp in calibration_components:
        gp = ComponentGP(**gp_kwargs)
        gp.fit(comp.time, comp.full_observation)

        mean, _ = gp.predict(dense_grid)

        rows.append(
            trajectory_features(
                dense_grid,
                mean,
            )
        )

    return ReferenceBaseline(
        np.vstack(rows)
    )


def calibrate_threshold(
    calibration_components,
    baseline,
    dense_grid,
    gp_kwargs=None,
    percentile=95.0,
):
    gp_kwargs = {} if gp_kwargs is None else gp_kwargs

    scores = []

    for comp in calibration_components:
        gp = ComponentGP(**gp_kwargs)
        gp.fit(
            comp.time,
            comp.full_observation,
        )

        mean, _ = gp.predict(dense_grid)

        features = trajectory_features(
            dense_grid,
            mean,
        )

        scores.append(
            baseline.score(features)
        )

    return float(
        np.percentile(
            scores,
            percentile,
        )
    )


def fixed_rate_classification(
    component,
    sample_indices,
    dense_grid,
    baseline,
    threshold,
    gp_kwargs=None,
):
    gp_kwargs = {} if gp_kwargs is None else gp_kwargs

    gp = ComponentGP(**gp_kwargs)

    gp.fit(
        component.time[sample_indices],
        component.full_observation[sample_indices],
    )

    mean, _ = gp.predict(dense_grid)

    features = trajectory_features(
        dense_grid,
        mean,
    )

    score = baseline.score(features)

    return {
        "decision": classify(
            score,
            threshold,
        ),
        "score": float(score),
        "features": features.tolist(),
        "measurement_count": int(
            len(sample_indices)
        ),
    }


def adaptive_classification(
    component,
    dense_grid,
    baseline,
    threshold,
    base_interval=0.05,
    max_interval=0.50,
    p_threshold=0.05,
    n_samples=50,
    gp_kwargs=None,
    seed=0,
    minimum_evidence_fraction=0.25,
):
    """
    Adaptive FALCON screening.

    The scheduler does not treat early GP extrapolation as
    sufficient evidence for adaptive spacing decisions.

    Until the configured fraction of the observation window
    has actually been observed, measurements remain frequent.

    After sufficient evidence is available, posterior
    decision-instability controls the measurement interval.

    Final classification is performed using the complete
    selected measurement set.
    """

    gp_kwargs = {} if gp_kwargs is None else gp_kwargs

    full_t = np.asarray(
        component.time,
        dtype=float,
    )

    full_y = np.asarray(
        component.full_observation,
        dtype=float,
    )

    if len(full_t) == 0:
        raise ValueError(
            "Component contains no measurements."
        )

    if len(full_t) != len(full_y):
        raise ValueError(
            "Time and observation arrays must have "
            "the same length."
        )

    if len(full_t) == 1:
        return {
            "decision": "NORMAL",
            "score": 0.0,
            "features": [],
            "measurement_count": 1,
            "selected_indices": [0],
            "selected_times": [
                float(full_t[0])
            ],
            "schedule_log": [],
        }

    selected = [0]

    next_t = float(
        full_t[0] + base_interval
    )

    logs = []

    while next_t <= full_t[-1] + 1e-9:

        idx = int(
            np.argmin(
                np.abs(full_t - next_t)
            )
        )

        if idx <= selected[-1]:
            idx = selected[-1] + 1

        if idx >= len(full_t):
            break

        selected.append(idx)

        gp = ComponentGP(**gp_kwargs)

        gp.fit(
            full_t[selected],
            full_y[selected],
        )

        observed_time = float(
            full_t[selected[-1]]
        )

        enough_evidence = evidence_sufficient(
            observed_time=observed_time,
            full_time=full_t,
            minimum_fraction=minimum_evidence_fraction,
        )

        if not enough_evidence:

            interval = base_interval

            majority = "INSUFFICIENT_EVIDENCE"
            instability = None

        else:

            instability, majority, _ = (
                decision_instability(
                    gp,
                    dense_grid,
                    baseline,
                    threshold,
                    n_samples=n_samples,
                    seed=seed + len(selected),
                )
            )

            interval = next_interval(
                instability,
                base_interval,
                max_interval,
                p_threshold,
                growth=2.0,
            )

        logs.append(
            {
                "time": observed_time,
                "measurements": len(selected),
                "evidence_fraction": (
                    (observed_time - full_t[0])
                    / max(
                        full_t[-1] - full_t[0],
                        1e-12,
                    )
                ),
                "evidence_sufficient": (
                    bool(enough_evidence)
                ),
                "decision_instability": (
                    None
                    if instability is None
                    else float(instability)
                ),
                "majority_decision": majority,
                "next_interval": float(interval),
            }
        )

        next_t = float(
            full_t[idx] + interval
        )

    # Final FALCON classification.
    #
    # This intentionally uses all selected observations
    # collected by the adaptive schedule.
    gp = ComponentGP(**gp_kwargs)

    gp.fit(
        full_t[selected],
        full_y[selected],
    )

    mean, _ = gp.predict(dense_grid)

    features = trajectory_features(
        dense_grid,
        mean,
    )

    score = baseline.score(
        features
    )

    return {
        "decision": classify(
            score,
            threshold,
        ),
        "score": float(score),
        "features": features.tolist(),
        "measurement_count": int(
            len(selected)
        ),
        "selected_indices": [
            int(i)
            for i in selected
        ],
        "selected_times": [
            float(t)
            for t in full_t[selected]
        ],
        "schedule_log": logs,
    }
