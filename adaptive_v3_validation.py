from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from falcon.gp_model import ComponentGP
from falcon.features import trajectory_features
from falcon.pipeline import build_reference, calibrate_threshold
from falcon.simulator import make_calibration_set, make_component


# ============================================================
# FALCON ADAPTIVE V3 VALIDATION
#
# Purpose:
# Compare:
#   1. Fixed-rate sampling + V3 classifier
#   2. Adaptive sampling + V3 classifier
#
# V3 classifier:
#   Mahalanobis anomaly
#           AND
#   degradation-direction anomaly
#
# IMPORTANT:
# This experiment uses the synthetic RC proxy.
# It does NOT represent real semiconductor physics.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

TIME = np.linspace(0.0, 5.0, 101)

DENSE_GRID = np.linspace(
    0.0,
    5.0,
    501,
)

SEEDS = list(range(10))

NOISE_STD = 0.015

# Synthetic test groups
NORMAL_TAUS = [
    0.98,
    1.00,
    1.02,
]

DEGRADATION_TAUS = [
    0.60,
    0.70,
    0.80,
    0.85,
    0.88,
    0.90,
]

WRONG_DIRECTION_TAUS = [
    1.04,
    1.06,
    1.08,
    1.10,
]

# Normal-only calibration
THRESHOLD_PERCENTILE = 95.0

# Adaptive scheduler settings
BASE_INTERVAL = 0.05
MAX_INTERVAL = 0.50

INSTABILITY_THRESHOLDS = [
    0.04,
    0.10,
    0.20,
]

N_POSTERIOR = 50

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    exist_ok=True
)

GP_KWARGS = {
    "length_scale": 0.75,
    "signal_variance": 4.0,
    "noise_variance": NOISE_STD ** 2,
    "random_state": 0,
}


# ============================================================
# GP + FEATURE EXTRACTION
# ============================================================

def extract_features(
    time,
    values,
):
    """
    Fit GP to the available observations and extract:

        [AUC, tau63]

    The GP is only used as the trajectory model.
    """

    time = np.asarray(
        time,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    gp = ComponentGP(
        **GP_KWARGS
    )

    gp.fit(
        time,
        values,
    )

    mean, _ = gp.predict(
        DENSE_GRID
    )

    return trajectory_features(
        DENSE_GRID,
        mean,
    )


# ============================================================
# DIRECTIONAL REFERENCE
# ============================================================

def robust_scale(values):
    """
    Robust scale using MAD.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    median = np.median(
        values
    )

    mad = np.median(
        np.abs(
            values - median
        )
    )

    scale = 1.4826 * mad

    if scale < 1e-9:
        scale = np.std(
            values
        )

    if scale < 1e-9:
        scale = 1e-9

    return float(scale)


def build_directional_reference(
    calibration,
):
    """
    Build a NORMAL reference from calibration components.
    """

    feature_matrix = []

    for component in calibration:

        features = extract_features(
            component.time,
            component.full_observation,
        )

        feature_matrix.append(
            features
        )

    feature_matrix = np.asarray(
        feature_matrix,
        dtype=float,
    )

    median = np.median(
        feature_matrix,
        axis=0,
    )

    scales = np.array(
        [
            robust_scale(
                feature_matrix[:, 0]
            ),
            robust_scale(
                feature_matrix[:, 1]
            ),
        ],
        dtype=float,
    )

    return {
        "features": feature_matrix,
        "median": median,
        "scales": scales,
    }


# ============================================================
# DIRECTIONAL SCORE
# ============================================================

def directional_score(
    features,
    reference,
):
    """
    Current RC-proxy assumption:

        lower AUC   = degradation direction
        lower tau63 = degradation direction

    Only deviation toward that direction contributes.

    This function must NOT be interpreted as universal
    semiconductor degradation physics.
    """

    features = np.asarray(
        features,
        dtype=float,
    )

    center = reference[
        "median"
    ]

    scales = reference[
        "scales"
    ]

    auc_direction = (
        center[0]
        - features[0]
    ) / scales[0]

    tau_direction = (
        center[1]
        - features[1]
    ) / scales[1]

    auc_positive = max(
        0.0,
        auc_direction,
    )

    tau_positive = max(
        0.0,
        tau_direction,
    )

    score = np.sqrt(
        auc_positive ** 2
        +
        tau_positive ** 2
    )

    return float(
        score
    )


# ============================================================
# DIRECTIONAL THRESHOLD
# ============================================================

def calibrate_directional_threshold(
    calibration,
    directional_reference,
):
    """
    Threshold is learned ONLY from NORMAL calibration data.
    """

    scores = []

    for component in calibration:

        features = extract_features(
            component.time,
            component.full_observation,
        )

        score = directional_score(
            features,
            directional_reference,
        )

        scores.append(
            score
        )

    return float(
        np.percentile(
            np.asarray(scores),
            THRESHOLD_PERCENTILE,
        )
    )


# ============================================================
# V3 CLASSIFIER
# ============================================================

def v3_decision(
    features,
    baseline,
    mahalanobis_threshold,
    directional_reference,
    directional_threshold,
):
    """
    V3 candidate classifier:

        Mahalanobis abnormal
                  AND
        degradation-direction evidence

    """

    mahalanobis = baseline.score(
        features
    )

    directional = directional_score(
        features,
        directional_reference,
    )

    mahalanobis_anomaly = (
        mahalanobis
        >
        mahalanobis_threshold
    )

    directional_anomaly = (
        directional
        >
        directional_threshold
    )

    # Candidate V3 rule
    anomaly = (
        mahalanobis_anomaly
        and
        directional_anomaly
    )

    decision = (
        "ANOMALOUS"
        if anomaly
        else
        "NORMAL"
    )

    return {
        "decision": decision,
        "mahalanobis": float(
            mahalanobis
        ),
        "directional": float(
            directional
        ),
    }


# ============================================================
# FIXED-RATE CLASSIFICATION
# ============================================================

def fixed_rate_classification(
    component,
    baseline,
    mahalanobis_threshold,
    directional_reference,
    directional_threshold,
):
    """
    Uses all 101 available observations.
    """

    features = extract_features(
        component.time,
        component.full_observation,
    )

    result = v3_decision(
        features,
        baseline,
        mahalanobis_threshold,
        directional_reference,
        directional_threshold,
    )

    result["measurement_count"] = len(
        component.time
    )

    result["features"] = features

    return result


# ============================================================
# DECISION INSTABILITY
# ============================================================

def decision_instability(
    gp,
    baseline,
    mahalanobis_threshold,
    directional_reference,
    directional_threshold,
    seed,
):
    """
    Sample posterior trajectories from the current GP.

    Each trajectory is passed through the SAME V3 classifier.

    Instability =
        fraction of posterior decisions that disagree
        with the majority decision.
    """

    samples = gp.sample_trajectories(
        DENSE_GRID,
        n_samples=N_POSTERIOR,
        rng_seed=seed,
    )

    decisions = []

    for trajectory in samples:

        features = trajectory_features(
            DENSE_GRID,
            trajectory,
        )

        result = v3_decision(
            features,
            baseline,
            mahalanobis_threshold,
            directional_reference,
            directional_threshold,
        )

        decisions.append(
            result["decision"]
        )

    decisions = np.asarray(
        decisions
    )

    normal_count = np.sum(
        decisions == "NORMAL"
    )

    anomaly_count = np.sum(
        decisions == "ANOMALOUS"
    )

    if anomaly_count > normal_count:
        majority = "ANOMALOUS"
    else:
        majority = "NORMAL"

    instability = np.mean(
        decisions != majority
    )

    return (
        float(instability),
        majority,
    )


# ============================================================
# ADAPTIVE CLASSIFICATION
# ============================================================

def adaptive_classification(
    component,
    baseline,
    mahalanobis_threshold,
    directional_reference,
    directional_threshold,
    instability_threshold,
    seed,
):
    """
    Adaptive measurement process.

    Start with first observation.

    After each measurement:
        GP
        ↓
        posterior trajectories
        ↓
        V3 decision instability
        ↓
        choose next interval
    """

    full_t = np.asarray(
        component.time,
        dtype=float,
    )

    full_y = np.asarray(
        component.full_observation,
        dtype=float,
    )

    selected = [0]

    next_t = (
        full_t[0]
        +
        BASE_INTERVAL
    )

    schedule_log = []

    while (
        next_t
        <=
        full_t[-1]
        +
        1e-9
    ):

        # Find closest available measurement
        idx = int(
            np.argmin(
                np.abs(
                    full_t
                    -
                    next_t
                )
            )
        )

        # Guarantee forward progress
        if idx <= selected[-1]:

            idx = (
                selected[-1]
                +
                1
            )

        if idx >= len(
            full_t
        ):
            break

        selected.append(
            idx
        )

        current_t = full_t[
            selected
        ]

        current_y = full_y[
            selected
        ]

        # Fit GP to currently observed measurements
        gp = ComponentGP(
            **GP_KWARGS
        )

        gp.fit(
            current_t,
            current_y,
        )

        instability, majority = (
            decision_instability(
                gp,
                baseline,
                mahalanobis_threshold,
                directional_reference,
                directional_threshold,
                seed
                +
                len(selected),
            )
        )

        # ----------------------------------------------------
        # Scheduling rule
        #
        # High instability:
        #     measure again soon.
        #
        # Low instability:
        #     measurement interval can expand.
        # ----------------------------------------------------

        if (
            instability
            >
            instability_threshold
        ):

            interval = BASE_INTERVAL

        else:

            interval = min(
                MAX_INTERVAL,
                BASE_INTERVAL * 2.0,
            )

        schedule_log.append({
            "time": float(
                full_t[idx]
            ),
            "measurements": len(
                selected
            ),
            "instability": float(
                instability
            ),
            "majority": majority,
            "next_interval": float(
                interval
            ),
        })

        next_t = (
            full_t[idx]
            +
            interval
        )

    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    final_t = full_t[
        selected
    ]

    final_y = full_y[
        selected
    ]

    final_features = extract_features(
        final_t,
        final_y,
    )

    final_result = v3_decision(
        final_features,
        baseline,
        mahalanobis_threshold,
        directional_reference,
        directional_threshold,
    )

    return {
        "decision":
            final_result["decision"],

        "measurement_count":
            len(selected),

        "selected_indices":
            selected,

        "selected_times":
            full_t[selected].tolist(),

        "features":
            final_features,

        "schedule_log":
            schedule_log,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("FALCON ADAPTIVE V3 VALIDATION")
    print("=" * 78)
    print()

    print(
        "Fixed-rate = all 101 measurements"
    )

    print(
        "Adaptive   = FALCON adaptive schedule"
    )

    print(
        "Classifier = Mahalanobis AND directional"
    )

    print()

    all_rows = []

    # ========================================================
    # SEEDS
    # ========================================================

    for seed_index, seed in enumerate(
        SEEDS,
        start=1,
    ):

        print(
            f"Running seed "
            f"{seed_index}/{len(SEEDS)}..."
        )

        calibration_rng = (
            np.random.default_rng(
                seed
            )
        )

        calibration = make_calibration_set(
            n_normal=30,
            time=TIME,
            rng=calibration_rng,
        )

        # ----------------------------------------------------
        # Mahalanobis reference
        # ----------------------------------------------------

        baseline = build_reference(
            calibration,
            DENSE_GRID,
            gp_kwargs=GP_KWARGS,
        )

        mahalanobis_threshold = (
            calibrate_threshold(
                calibration,
                baseline,
                DENSE_GRID,
                gp_kwargs=GP_KWARGS,
                percentile=THRESHOLD_PERCENTILE,
            )
        )

        # ----------------------------------------------------
        # Directional reference
        # ----------------------------------------------------

        directional_reference = (
            build_directional_reference(
                calibration
            )
        )

        directional_threshold = (
            calibrate_directional_threshold(
                calibration,
                directional_reference,
            )
        )

        # ====================================================
        # TEST GROUPS
        # ====================================================

        groups = [
            (
                "CLEAR_NORMAL",
                NORMAL_TAUS,
            ),
            (
                "DEGRADATION",
                DEGRADATION_TAUS,
            ),
            (
                "WRONG_DIRECTION",
                WRONG_DIRECTION_TAUS,
            ),
        ]

        for group_name, tau_values in groups:

            for tau in tau_values:

                # Deterministic but different noise
                test_rng = (
                    np.random.default_rng(
                        seed * 100000
                        +
                        int(
                            round(
                                tau * 1000
                            )
                        )
                    )
                )

                component = make_component(
                    component_id=(
                        f"{group_name}_"
                        f"tau{tau:.2f}_"
                        f"seed{seed}"
                    ),
                    time=TIME,
                    tau=tau,
                    noise_std=NOISE_STD,
                    label=group_name,
                    rng=test_rng,
                )

                # ------------------------------------------------
                # Fixed-rate
                # ------------------------------------------------

                fixed = (
                    fixed_rate_classification(
                        component,
                        baseline,
                        mahalanobis_threshold,
                        directional_reference,
                        directional_threshold,
                    )
                )

                # ------------------------------------------------
                # Adaptive for each instability threshold
                # ------------------------------------------------

                for instability_threshold in (
                    INSTABILITY_THRESHOLDS
                ):

                    adaptive = (
                        adaptive_classification(
                            component,
                            baseline,
                            mahalanobis_threshold,
                            directional_reference,
                            directional_threshold,
                            instability_threshold,
                            seed,
                        )
                    )

                    fixed_count = (
                        fixed[
                            "measurement_count"
                        ]
                    )

                    adaptive_count = (
                        adaptive[
                            "measurement_count"
                        ]
                    )

                    measurement_reduction = (
                        1.0
                        -
                        (
                            adaptive_count
                            /
                            fixed_count
                        )
                    )

                    agreement = (
                        fixed["decision"]
                        ==
                        adaptive["decision"]
                    )

                    all_rows.append({
                        "seed":
                            seed,

                        "group":
                            group_name,

                        "tau":
                            tau,

                        "instability_threshold":
                            instability_threshold,

                        "fixed_decision":
                            fixed["decision"],

                        "adaptive_decision":
                            adaptive["decision"],

                        "fixed_measurements":
                            fixed_count,

                        "adaptive_measurements":
                            adaptive_count,

                        "measurement_reduction":
                            float(
                                measurement_reduction
                            ),

                        "agreement":
                            int(
                                agreement
                            ),
                    })

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_path = (
        RESULTS_DIR
        /
        "adaptive_v3_validation_raw.csv"
    )

    fields = [
        "seed",
        "group",
        "tau",
        "instability_threshold",
        "fixed_decision",
        "adaptive_decision",
        "fixed_measurements",
        "adaptive_measurements",
        "measurement_reduction",
        "agreement",
    ]

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    # ========================================================
    # GROUP RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print("ADAPTIVE SAMPLING RESULTS")
    print("=" * 78)

    summary_rows = []

    for threshold in (
        INSTABILITY_THRESHOLDS
    ):

        print()
        print(
            f"INSTABILITY THRESHOLD = "
            f"{threshold:.2f}"
        )

        threshold_rows = [
            row
            for row in all_rows
            if np.isclose(
                row[
                    "instability_threshold"
                ],
                threshold,
            )
        ]

        for group in [
            "CLEAR_NORMAL",
            "DEGRADATION",
            "WRONG_DIRECTION",
        ]:

            group_rows = [
                row
                for row in threshold_rows
                if row["group"]
                ==
                group
            ]

            mean_reduction = np.mean([
                row[
                    "measurement_reduction"
                ]
                for row in group_rows
            ])

            agreement = np.mean([
                row[
                    "agreement"
                ]
                for row in group_rows
            ])

            print(
                f"{group:<20}"
                f" reduction="
                f"{100 * mean_reduction:7.2f}%"
                f"  agreement="
                f"{100 * agreement:7.2f}%"
            )

            summary_rows.append({
                "threshold":
                    threshold,

                "group":
                    group,

                "mean_reduction":
                    float(
                        mean_reduction
                    ),

                "agreement":
                    float(
                        agreement
                    ),
            })

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print("OVERALL RESULTS")
    print("=" * 78)

    for threshold in (
        INSTABILITY_THRESHOLDS
    ):

        threshold_rows = [
            row
            for row in all_rows
            if np.isclose(
                row[
                    "instability_threshold"
                ],
                threshold,
            )
        ]

        mean_reduction = np.mean([
            row[
                "measurement_reduction"
            ]
            for row in threshold_rows
        ])

        agreement = np.mean([
            row[
                "agreement"
            ]
            for row in threshold_rows
        ])

        print(
            f"p={threshold:.2f}"
            f"  reduction="
            f"{100 * mean_reduction:.2f}%"
            f"  agreement="
            f"{100 * agreement:.2f}%"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        RESULTS_DIR
        /
        "adaptive_v3_validation_summary.csv"
    )

    summary_fields = [
        "threshold",
        "group",
        "mean_reduction",
        "agreement",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 78)
    print("FILES")
    print("=" * 78)

    print(
        f"Raw results     : "
        f"{raw_path}"
    )

    print(
        f"Summary results : "
        f"{summary_path}"
    )

    print()
    print("=" * 78)
    print("INTERPRETATION RULE")
    print("=" * 78)

    print(
        "High agreement + useful measurement reduction "
        "supports the adaptive-sampling concept."
    )

    print(
        "If agreement drops substantially, the scheduler "
        "needs redesign before being used in FALCON."
    )

    print(
        "This experiment does NOT prove real semiconductor "
        "degradation detection."
    )

    print(
        "The degradation direction is specific to the "
        "synthetic RC proxy."
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
