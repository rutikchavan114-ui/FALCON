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

# Fixed operating point for this robustness experiment.
# Do NOT optimize this threshold during this experiment.
INSTABILITY_THRESHOLD = 0.05

SEEDS = range(10)

NOISE_LEVELS = [
    0.005,
    0.015,
    0.030,
    0.050,
]


def make_robust_test_set(time, rng, noise_std):
    """
    Synthetic robustness test set.

    2 NORMAL
    2 BORDERLINE
    2 ANOMALOUS

    Borderline components are treated as NORMAL
    for binary screening accuracy.
    """

    components = []

    cases = [
        ("NORMAL", 1.00),
        ("NORMAL", 1.03),
        ("BORDERLINE", 0.90),
        ("BORDERLINE", 0.85),
        ("ANOMALOUS", 0.70),
        ("ANOMALOUS", 0.60),
    ]

    for i, (label, tau) in enumerate(cases):

        latent = (
            5.0
            * np.exp(-time / tau)
        )

        observed = (
            latent
            + rng.normal(
                0.0,
                noise_std,
                size=len(time),
            )
        )

        components.append(
            ComponentTrajectory(
                component_id=f"ROBUST-{i + 1:02d}",
                time=time.copy(),
                latent=latent,
                full_observation=observed,
                tau=tau,
                label=label,
            )
        )

    return components


def evaluate(seed, noise_std):

    # --------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------

    calibration = make_calibration_set(
        n_normal=30,
        time=TIME,
        rng=np.random.default_rng(seed),
    )

    # --------------------------------------------------
    # TEST SET
    # --------------------------------------------------

    test = make_robust_test_set(
        TIME,
        np.random.default_rng(seed + 1000),
        noise_std,
    )

    # --------------------------------------------------
    # NORMAL REFERENCE
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
    # FIXED-RATE VS FALCON
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
            p_threshold=INSTABILITY_THRESHOLD,
            minimum_evidence_fraction=0.25,
        )

        fixed_results.append(fixed)
        adaptive_results.append(adaptive)

    # --------------------------------------------------
    # MEASUREMENT REDUCTION
    # --------------------------------------------------

    fixed_measurements = sum(
        result["measurement_count"]
        for result in fixed_results
    )

    adaptive_measurements = sum(
        result["measurement_count"]
        for result in adaptive_results
    )

    reduction = (
        100.0
        * (
            1.0
            - adaptive_measurements
            / fixed_measurements
        )
    )

    # --------------------------------------------------
    # DECISION AGREEMENT
    # --------------------------------------------------

    agreement = (
        sum(
            fixed["decision"]
            == adaptive["decision"]
            for fixed, adaptive in zip(
                fixed_results,
                adaptive_results,
            )
        )
        / len(test)
    )

    # --------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------

    # Binary screening:
    # NORMAL + BORDERLINE -> NORMAL
    # ANOMALOUS -> ANOMALOUS

    truth_binary = [
        (
            "ANOMALOUS"
            if component.label == "ANOMALOUS"
            else "NORMAL"
        )
        for component in test
    ]

    # --------------------------------------------------
    # ACCURACY
    # --------------------------------------------------

    fixed_accuracy = (
        sum(
            result["decision"] == actual
            for result, actual in zip(
                fixed_results,
                truth_binary,
            )
        )
        / len(test)
    )

    adaptive_accuracy = (
        sum(
            result["decision"] == actual
            for result, actual in zip(
                adaptive_results,
                truth_binary,
            )
        )
        / len(test)
    )

    # --------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------

    anomalous_count = sum(
        actual == "ANOMALOUS"
        for actual in truth_binary
    )

    normal_count = sum(
        actual == "NORMAL"
        for actual in truth_binary
    )

    # --------------------------------------------------
    # FALSE NEGATIVE / FALSE POSITIVE
    # --------------------------------------------------

    fixed_fn = 0
    fixed_fp = 0

    adaptive_fn = 0
    adaptive_fp = 0

    for actual, fixed, adaptive in zip(
        truth_binary,
        fixed_results,
        adaptive_results,
    ):

        # Fixed false negative
        if (
            actual == "ANOMALOUS"
            and fixed["decision"] == "NORMAL"
        ):
            fixed_fn += 1

        # Fixed false positive
        if (
            actual == "NORMAL"
            and fixed["decision"] == "ANOMALOUS"
        ):
            fixed_fp += 1

        # FALCON false negative
        if (
            actual == "ANOMALOUS"
            and adaptive["decision"] == "NORMAL"
        ):
            adaptive_fn += 1

        # FALCON false positive
        if (
            actual == "NORMAL"
            and adaptive["decision"] == "ANOMALOUS"
        ):
            adaptive_fp += 1

    # --------------------------------------------------
    # ANOMALY DETECTION RATE
    # --------------------------------------------------

    fixed_anomaly_detection_rate = (
        (anomalous_count - fixed_fn)
        / anomalous_count
    )

    adaptive_anomaly_detection_rate = (
        (anomalous_count - adaptive_fn)
        / anomalous_count
    )

    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------

    return {
        "reduction": reduction,
        "agreement": agreement,

        "fixed_accuracy": fixed_accuracy,
        "adaptive_accuracy": adaptive_accuracy,

        "fixed_fn_rate": (
            fixed_fn / anomalous_count
        ),
        "adaptive_fn_rate": (
            adaptive_fn / anomalous_count
        ),

        "fixed_fp_rate": (
            fixed_fp / normal_count
        ),
        "adaptive_fp_rate": (
            adaptive_fp / normal_count
        ),

        "fixed_anomaly_detection_rate": (
            fixed_anomaly_detection_rate
        ),
        "adaptive_anomaly_detection_rate": (
            adaptive_anomaly_detection_rate
        ),
    }


def main():

    print("FALCON ROBUSTNESS EVALUATION")
    print("=" * 60)

    print(
        f"Instability threshold: "
        f"{INSTABILITY_THRESHOLD:.2f}"
    )

    print(
        "Seeds: 10"
    )

    print(
        "Test cases per seed: 6 "
        "(2 normal, 2 borderline, 2 anomalous)"
    )

    for noise_std in NOISE_LEVELS:

        results = [
            evaluate(
                seed,
                noise_std,
            )
            for seed in SEEDS
        ]

        print()
        print("=" * 60)
        print(
            f"Noise standard deviation: "
            f"{noise_std:.3f}"
        )
        print("=" * 60)

        print(
            f"Mean measurement reduction: "
            f"{np.mean([r['reduction'] for r in results]):.2f}%"
        )

        print(
            f"Mean decision agreement: "
            f"{100 * np.mean([r['agreement'] for r in results]):.1f}%"
        )

        print(
            f"Fixed accuracy: "
            f"{100 * np.mean([r['fixed_accuracy'] for r in results]):.1f}%"
        )

        print(
            f"FALCON accuracy: "
            f"{100 * np.mean([r['adaptive_accuracy'] for r in results]):.1f}%"
        )

        print(
            f"Fixed anomaly detection rate: "
            f"{100 * np.mean([r['fixed_anomaly_detection_rate'] for r in results]):.1f}%"
        )

        print(
            f"FALCON anomaly detection rate: "
            f"{100 * np.mean([r['adaptive_anomaly_detection_rate'] for r in results]):.1f}%"
        )

        print(
            f"Fixed false-negative rate: "
            f"{100 * np.mean([r['fixed_fn_rate'] for r in results]):.1f}%"
        )

        print(
            f"FALCON false-negative rate: "
            f"{100 * np.mean([r['adaptive_fn_rate'] for r in results]):.1f}%"
        )

        print(
            f"Fixed false-positive rate: "
            f"{100 * np.mean([r['fixed_fp_rate'] for r in results]):.1f}%"
        )

        print(
            f"FALCON false-positive rate: "
            f"{100 * np.mean([r['adaptive_fp_rate'] for r in results]):.1f}%"
        )


if __name__ == "__main__":
    main()
