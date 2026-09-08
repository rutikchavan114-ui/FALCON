from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from falcon.gp_model import ComponentGP
from falcon.features import trajectory_features
from falcon.pipeline import build_reference, calibrate_threshold
from falcon.simulator import make_calibration_set, make_component
from falcon.anomaly import classify


# ============================================================
# CONFIG
# ============================================================

TIME = np.linspace(0.0, 5.0, 101)
DENSE_GRID = np.linspace(0.0, 5.0, 501)

SEEDS = list(range(10))

NOISE_STD = 0.015

NORMAL_TAUS = [0.98, 1.00, 1.02]
DEGRADATION_TAUS = [0.60, 0.70, 0.80, 0.85, 0.88, 0.90]
WRONG_DIRECTION_TAUS = [1.04, 1.06, 1.08, 1.10]

THRESHOLD_PERCENTILE = 95.0

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

GP_KWARGS = {
    "length_scale": 0.75,
    "signal_variance": 4.0,
    "noise_variance": NOISE_STD ** 2,
    "random_state": 0,
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(component):

    gp = ComponentGP(**GP_KWARGS)

    gp.fit(
        component.time,
        component.full_observation,
    )

    mean, _ = gp.predict(DENSE_GRID)

    return trajectory_features(
        DENSE_GRID,
        mean,
    )


# ============================================================
# ROBUST SCALE
# ============================================================

def robust_scale(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    scale = 1.4826 * mad

    if scale < 1e-9:
        scale = np.std(values)

    if scale < 1e-9:
        scale = 1e-9

    return float(scale)


# ============================================================
# DIRECTIONAL REFERENCE
# ============================================================

def build_directional_reference(calibration):

    features = np.asarray(
        [
            extract_features(c)
            for c in calibration
        ],
        dtype=float,
    )

    median = np.median(
        features,
        axis=0,
    )

    scales = np.array(
        [
            robust_scale(features[:, 0]),
            robust_scale(features[:, 1]),
        ]
    )

    return {
        "features": features,
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

    features = np.asarray(
        features,
        dtype=float,
    )

    center = reference["median"]
    scales = reference["scales"]

    # Current RC proxy:
    # lower AUC + lower tau63 = degradation direction

    auc_direction = (
        center[0] - features[0]
    ) / scales[0]

    tau_direction = (
        center[1] - features[1]
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

    return (
        float(score),
        float(auc_direction),
        float(tau_direction),
    )


# ============================================================
# CALIBRATE DIRECTIONAL THRESHOLD
# ============================================================

def calibrate_directional_threshold(
    calibration,
    reference,
):

    scores = []

    for component in calibration:

        features = extract_features(
            component
        )

        score, _, _ = directional_score(
            features,
            reference,
        )

        scores.append(score)

    return float(
        np.percentile(
            scores,
            THRESHOLD_PERCENTILE,
        )
    )


# ============================================================
# CLASSIFIER
# ============================================================

def classify_three_models(
    mahalanobis,
    mahal_threshold,
    directional,
    directional_threshold,
):

    model_a = (
        mahalanobis
        > mahal_threshold
    )

    model_b = (
        directional
        > directional_threshold
    )

    # IMPORTANT:
    # Both conditions must agree.
    model_c = (
        model_a
        and
        model_b
    )

    return (
        model_a,
        model_b,
        model_c,
    )


# ============================================================
# EVALUATE ONE COMPONENT
# ============================================================

def evaluate_component(
    component,
    baseline,
    mahal_threshold,
    directional_reference,
    directional_threshold,
):

    features = extract_features(
        component
    )

    mahalanobis = baseline.score(
        features
    )

    directional, auc_dir, tau_dir = (
        directional_score(
            features,
            directional_reference,
        )
    )

    model_a, model_b, model_c = (
        classify_three_models(
            mahalanobis,
            mahal_threshold,
            directional,
            directional_threshold,
        )
    )

    return {
        "tau": component.tau,
        "auc": float(features[0]),
        "tau63": float(features[1]),
        "mahalanobis": float(mahalanobis),
        "directional": float(directional),
        "auc_direction": float(auc_dir),
        "tau_direction": float(tau_dir),
        "model_a": int(model_a),
        "model_b": int(model_b),
        "model_c": int(model_c),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)
    print("FALCON CLASSIFIER V3")
    print("MAHALANOBIS vs DIRECTIONAL vs AND")
    print("=" * 76)
    print()

    all_rows = []

    for seed in SEEDS:

        print(
            f"Running seed {seed + 1}/{len(SEEDS)}..."
        )

        rng = np.random.default_rng(
            seed
        )

        calibration = make_calibration_set(
            n_normal=30,
            time=TIME,
            rng=rng,
        )

        # --------------------------------------------
        # Mahalanobis baseline
        # --------------------------------------------

        baseline = build_reference(
            calibration,
            DENSE_GRID,
            gp_kwargs=GP_KWARGS,
        )

        mahal_threshold = calibrate_threshold(
            calibration,
            baseline,
            DENSE_GRID,
            gp_kwargs=GP_KWARGS,
            percentile=THRESHOLD_PERCENTILE,
        )

        # --------------------------------------------
        # Directional baseline
        # --------------------------------------------

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

        # --------------------------------------------
        # Test groups
        # --------------------------------------------

        test_groups = [
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

        for group_name, tau_values in test_groups:

            for tau in tau_values:

                test_rng = np.random.default_rng(
                    seed * 100000
                    + int(round(tau * 1000))
                )

                component = make_component(
                    component_id=(
                        f"{group_name}_"
                        f"{tau:.2f}_"
                        f"S{seed}"
                    ),
                    time=TIME,
                    tau=tau,
                    noise_std=NOISE_STD,
                    label=group_name,
                    rng=test_rng,
                )

                result = evaluate_component(
                    component,
                    baseline,
                    mahal_threshold,
                    directional_reference,
                    directional_threshold,
                )

                result["seed"] = seed
                result["group"] = group_name

                all_rows.append(
                    result
                )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_path = (
        RESULTS_DIR
        / "classifier_v3_raw.csv"
    )

    fields = [
        "seed",
        "group",
        "tau",
        "auc",
        "tau63",
        "mahalanobis",
        "directional",
        "auc_direction",
        "tau_direction",
        "model_a",
        "model_b",
        "model_c",
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
        writer.writerows(all_rows)

    # ========================================================
    # GROUP METRICS
    # ========================================================

    print()
    print("=" * 76)
    print("GROUP PERFORMANCE")
    print("=" * 76)
    print()

    groups = [
        "CLEAR_NORMAL",
        "DEGRADATION",
        "WRONG_DIRECTION",
    ]

    model_names = [
        ("MODEL A - MAHALANOBIS", "model_a"),
        ("MODEL B - DIRECTIONAL", "model_b"),
        ("MODEL C - AND", "model_c"),
    ]

    summary_rows = []

    for group in groups:

        rows = [
            r
            for r in all_rows
            if r["group"] == group
        ]

        print()
        print(group)
        print("-" * 76)

        for model_name, key in model_names:

            rate = np.mean([
                r[key]
                for r in rows
            ])

            print(
                f"{model_name:<30}"
                f"{100 * rate:>8.2f}% anomalous"
            )

            summary_rows.append({
                "group": group,
                "model": model_name,
                "anomaly_rate": float(rate),
            })

    # ========================================================
    # IMPORTANT METRICS
    # ========================================================

    print()
    print("=" * 76)
    print("KEY METRICS")
    print("=" * 76)

    for model_name, key in model_names:

        normal_rows = [
            r
            for r in all_rows
            if r["group"] == "CLEAR_NORMAL"
        ]

        degradation_rows = [
            r
            for r in all_rows
            if r["group"] == "DEGRADATION"
        ]

        wrong_rows = [
            r
            for r in all_rows
            if r["group"] == "WRONG_DIRECTION"
        ]

        normal_fp = np.mean([
            r[key]
            for r in normal_rows
        ])

        degradation_detection = np.mean([
            r[key]
            for r in degradation_rows
        ])

        wrong_direction_fp = np.mean([
            r[key]
            for r in wrong_rows
        ])

        print()
        print(model_name)
        print(
            f"  Clear-normal false positive : "
            f"{100 * normal_fp:.2f}%"
        )

        print(
            f"  Degradation detection      : "
            f"{100 * degradation_detection:.2f}%"
        )

        print(
            f"  Wrong-direction false pos.  : "
            f"{100 * wrong_direction_fp:.2f}%"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        RESULTS_DIR
        / "classifier_v3_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "model",
                "anomaly_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # PLOT
    # ========================================================

    group_positions = np.arange(
        len(groups)
    )

    width = 0.25

    plt.figure(
        figsize=(10, 6)
    )

    for i, (model_name, key) in enumerate(
        model_names
    ):

        rates = []

        for group in groups:

            rows = [
                r
                for r in all_rows
                if r["group"] == group
            ]

            rates.append(
                100
                * np.mean([
                    r[key]
                    for r in rows
                ])
            )

        plt.bar(
            group_positions
            + (i - 1) * width,
            rates,
            width,
            label=model_name,
        )

    plt.xticks(
        group_positions,
        groups,
    )

    plt.ylabel(
        "Classified ANOMALOUS (%)"
    )

    plt.title(
        "FALCON Classifier V3 Comparison"
    )

    plt.ylim(
        0,
        105,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        RESULTS_DIR
        / "classifier_v3_comparison.png"
    )

    plt.savefig(
        plot_path,
        dpi=180,
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 76)
    print("FILES")
    print("=" * 76)
    print()

    print(
        f"Raw CSV     : {raw_path}"
    )

    print(
        f"Summary CSV : {summary_path}"
    )

    print(
        f"Plot        : {plot_path}"
    )

    print()
    print("=" * 76)
    print("DONE")
    print("=" * 76)


if __name__ == "__main__":
    main()
