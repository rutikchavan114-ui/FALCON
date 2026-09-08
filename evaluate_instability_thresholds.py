import numpy as np

from falcon.simulator import (
    make_calibration_set,
    make_test_set,
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
    0.20,
]

SEEDS = range(10)


def evaluate(seed, instability_threshold):

    calibration = make_calibration_set(
        30,
        TIME,
        np.random.default_rng(seed),
    )

    test = make_test_set(
        4,
        1,
        TIME,
        np.random.default_rng(seed + 100),
    )

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

    for component in test:

        fixed = fixed_rate_classification(
            component,
            np.arange(len(component.time)),
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

    fixed_count = sum(
        x["measurement_count"]
        for x in fixed_results
    )

    adaptive_count = sum(
        x["measurement_count"]
        for x in adaptive_results
    )

    reduction = 100.0 * (
        1.0 - adaptive_count / fixed_count
    )

    agreement = sum(
        f["decision"] == a["decision"]
        for f, a in zip(
            fixed_results,
            adaptive_results,
        )
    ) / len(test)

    return reduction, agreement


def main():

    print("FALCON INSTABILITY THRESHOLD EVALUATION")
    print("=" * 48)

    for p in INSTABILITY_THRESHOLDS:

        reductions = []
        agreements = []

        for seed in SEEDS:

            reduction, agreement = evaluate(
                seed,
                p,
            )

            reductions.append(reduction)
            agreements.append(agreement)

        print()
        print(f"Instability threshold: {p:.2f}")
        print(
            f"Mean reduction: "
            f"{np.mean(reductions):.2f}%"
        )
        print(
            f"Mean agreement: "
            f"{100*np.mean(agreements):.1f}%"
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
