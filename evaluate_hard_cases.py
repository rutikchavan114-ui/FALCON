import numpy as np

from falcon.simulator import (
    ComponentTrajectory,
    make_calibration_set,
)

from falcon.pipeline import (
    build_reference,
    calibrate_threshold,
    adaptive_classification,
    fixed_rate_classification,
)


TIME = np.linspace(0, 5, 101)

INSTABILITY_THRESHOLDS = [
    0.02,
    0.04,
    0.06,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

SEEDS = range(10)


def make_hard_test_set(time, rng):
    """
    Harder synthetic test set.

    2 clearly normal components
    2 borderline components
    2 clearly anomalous components
    """

    components = []

    # --------------------------------------------------
    # NORMAL COMPONENTS
    # --------------------------------------------------

    for i in range(2):

        tau = max(
            0.85,
            rng.normal(1.0, 0.04)
        )

        latent = (
            5.0
            * np.exp(-time / tau)
        )

        observed = (
            latent
            + rng.normal(
                0.0,
                0.015,
                size=len(time),
            )
        )

        components.append(
            ComponentTrajectory(
                component_id=f"H-NORMAL-{i+1}",
                time=time.copy(),
                latent=latent,
                full_observation=observed,
                tau=tau,
                label="NORMAL",
            )
        )

    # --------------------------------------------------
    # BORDERLINE COMPONENTS
    # --------------------------------------------------

    borderline_taus = [
        0.90,
        0.85,
    ]

    for i, tau in enumerate(borderline_taus):

        latent = (
            5.0
            * np.exp(-time / tau)
        )

        observed = (
            latent
            + rng.normal(
                0.0,
                0.015,
                size=len(time),
            )
        )

        components.append(
            ComponentTrajectory(
                component_id=f"H-BORDER-{i+1}",
                time=time.copy(),
                latent=latent,
                full_observation=observed,
                tau=tau,
                label="BORDERLINE",
            )
        )

    # --------------------------------------------------
    # CLEARLY ANOMALOUS COMPONENTS
    # --------------------------------------------------

    anomalous_taus = [
        0.70,
        0.60,
    ]

    for i, tau in enumerate(anomalous_taus):

        latent = (
            5.0
            * np.exp(-time / tau)
        )

        observed = (
            latent
            + rng.normal(
                0.0,
                0.015,
                size=len(time),
            )
        )

        components.append(
            ComponentTrajectory(
                component_id=f"H-ANOM-{i+1}",
                time=time.copy(),
                latent=latent,
                full_observation=observed,
                tau=tau,
                label="ANOMALOUS",
            )
        )

    return components


def evaluate(seed, instability_threshold):

    # --------------------------------------------------
    # CALIBRATION DATA
    # --------------------------------------------------

    calibration = make_calibration_set(
        n_normal=30,
        time=TIME,
        rng=np.random.default_rng(seed),
    )

    # --------------------------------------------------
    # HARD TEST DATA
    # --------------------------------------------------

    test = make_hard_test_set(
        TIME,
        np.random.default_rng(seed + 100),
    )

    # --------------------------------------------------
    # BUILD NORMAL REFERENCE
    # --------------------------------------------------

    baseline = build_reference(
        calibration,
        TIME,
    )

    threshold = calibrate_threshold(
        calibration,
        baseline,
        TIME,
    )

    fixed_results = []
    adaptive_results = []

    # --------------------------------------------------
    # RUN FIXED + FALCON
    # --------------------------------------------------

    for component in test:

        fixed = fixed_rate_classification(
            component,
            np.arange(
                len(component.full_observation)
            ),
            TIME,
            baseline,
            threshold,
        )

        adaptive = adaptive_classification(
            component,
            TIME,
            baseline,
            threshold,
            p_threshold=instability_threshold,
            minimum_evidence_fraction=0.25,
        )

        fixed_results.append(fixed)
        adaptive_results.append(adaptive)

    # --------------------------------------------------
    # MEASUREMENT REDUCTION
    # --------------------------------------------------

    fixed_count = sum(
        result["measurement_count"]
        for result in fixed_results
    )

    adaptive_count = sum(
        result["measurement_count"]
        for result in adaptive_results
    )

    reduction = (
        100.0
        * (
            1.0
            - adaptive_count / fixed_count
        )
    )

    # --------------------------------------------------
    # AGREEMENT WITH FIXED-RATE
    # --------------------------------------------------

    agreement = (
        sum(
            fixed["decision"]
            == adaptive["decision"]
            for fixed, adaptive
            in zip(
                fixed_results,
                adaptive_results,
            )
        )
        / len(test)
    )

    # --------------------------------------------------
    # RETURN COMPLETE RESULTS
    # --------------------------------------------------

    return {
        "reduction": reduction,
        "agreement": agreement,
        "fixed_results": fixed_results,
        "adaptive_results": adaptive_results,
        "threshold": threshold,
        "test": test,
    }


def main():

    print(
        "FALCON HARD-CASE THRESHOLD EVALUATION"
    )

    print("=" * 55)

    for p in INSTABILITY_THRESHOLDS:

        reductions = []
        agreements = []

        for seed in SEEDS:

            result = evaluate(
                seed,
                p,
            )

            reductions.append(
                result["reduction"]
            )

            agreements.append(
                result["agreement"]
            )

        print()

        print(
            f"Instability threshold: {p:.2f}"
        )

        print(
            f"Mean reduction: "
            f"{np.mean(reductions):.2f}%"
        )

        print(
            f"Mean agreement: "
            f"{100 * np.mean(agreements):.1f}%"
        )

        print(
            f"Perfect runs: "
            f"{sum(x == 1.0 for x in agreements)}/10"
        )

        print(
            f"Min reduction: "
            f"{min(reductions):.2f}%"
        )

        print(
            f"Max reduction: "
            f"{max(reductions):.2f}%"
        )


if __name__ == "__main__":
    main()
