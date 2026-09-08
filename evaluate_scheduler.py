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
FRACTIONS = [0.25, 0.35, 0.50]
SEEDS = range(10)


def run_one(seed, fraction):
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
            minimum_evidence_fraction=fraction,
        )

        fixed_results.append(fixed)
        adaptive_results.append(adaptive)

    fixed_measurements = sum(
        result["measurement_count"]
        for result in fixed_results
    )

    adaptive_measurements = sum(
        result["measurement_count"]
        for result in adaptive_results
    )

    reduction = 100.0 * (
        1.0
        - adaptive_measurements / fixed_measurements
    )

    agreement = sum(
        fixed["decision"] == adaptive["decision"]
        for fixed, adaptive in zip(
            fixed_results,
            adaptive_results,
        )
    ) / len(test)

    return reduction, agreement


def main():
    print("FALCON MULTI-SEED SCHEDULER EVALUATION")
    print("=" * 42)

    for fraction in FRACTIONS:
        reductions = []
        agreements = []

        for seed in SEEDS:
            reduction, agreement = run_one(
                seed,
                fraction,
            )

            reductions.append(reduction)
            agreements.append(agreement)

        mean_reduction = np.mean(reductions)
        mean_agreement = 100.0 * np.mean(agreements)
        perfect_runs = sum(
            agreement == 1.0
            for agreement in agreements
        )

        print()
        print(f"Evidence fraction: {fraction:.2f}")
        print(
            f"Mean reduction:    "
            f"{mean_reduction:.2f}%"
        )
        print(
            f"Mean agreement:    "
            f"{mean_agreement:.1f}%"
        )
        print(
            f"100% agreement:    "
            f"{perfect_runs}/10 runs"
        )
        print(
            f"Min reduction:     "
            f"{min(reductions):.2f}%"
        )
        print(
            f"Max reduction:     "
            f"{max(reductions):.2f}%"
        )


if __name__ == "__main__":
    main()
