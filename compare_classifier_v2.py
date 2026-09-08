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
# CONFIGURATION
# ============================================================

TIME = np.linspace(0.0, 5.0, 101)
DENSE_GRID = np.linspace(0.0, 5.0, 501)

TAU_VALUES = [
    1.10,
    1.08,
    1.06,
    1.04,
    1.02,
    1.00,
    0.98,
    0.96,
    0.94,
    0.92,
    0.90,
    0.88,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
]

SEEDS = list(range(10))

NOISE_STD = 0.015

MAHALANOBIS_PERCENTILE = 95.0

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
    """
    Fit the existing FALCON GP and extract:
        [AUC, tau63]
    """

    gp = ComponentGP(**GP_KWARGS)

    gp.fit(
        component.time,
        component.full_observation,
    )

    mean, _ = gp.predict(DENSE_GRID)

    features = trajectory_features(
        DENSE_GRID,
        mean,
    )

    return features


# ============================================================
# ROBUST SCALE
# ============================================================

def robust_scale(values):
    """
    MAD-based robust scale.
    """

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
            extract_features(component)
            for component in calibration
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
        ],
        dtype=float,
    )

    return {
        "features": features,
        "median": median,
        "scales": scales,
    }


# ============================================================
# DIRECTIONAL SCORE
# ============================================================

def directional_score(features, reference):
    """
    Current RC-proxy assumption:

        lower AUC   -> degradation direction
        lower tau63 -> degradation direction

    Only positive movement in that direction contributes.
    """

    features = np.asarray(
        features,
        dtype=float,
    )

    center = reference["median"]
    scales = reference["scales"]

    auc_direction = (
        center[0] - features[0]
    ) / scales[0]

    tau63_direction = (
        center[1] - features[1]
    ) / scales[1]

    auc_positive = max(
        0.0,
        auc_direction,
    )

    tau63_positive = max(
        0.0,
        tau63_direction,
    )

    score = np.sqrt(
        auc_positive ** 2
        +
        tau63_positive ** 2
    )

    return (
        float(score),
        float(auc_direction),
        float(tau63_direction),
    )


# ============================================================
# DIRECTIONAL THRESHOLD
# ============================================================

def calibrate_directional_threshold(
    calibration,
    directional_reference,
    percentile=95.0,
):
    """
    IMPORTANT:

    The directional threshold is learned ONLY from NORMAL
    calibration components.

    This avoids choosing an arbitrary threshold from the
    test tau sweep.
    """

    scores = []

    for component in calibration:

        features = extract_features(
            component
        )

        score, _, _ = directional_score(
            features,
            directional_reference,
        )

        scores.append(score)

    return float(
        np.percentile(
            np.asarray(scores),
            percentile,
        )
    )


# ============================================================
# COMBINED DECISION
# ============================================================

def combined_decision(
    mahalanobis_score,
    mahalanobis_threshold,
    directional,
    directional_threshold,
):
    """
    Candidate combined rule.

    A trajectory is anomalous if:

        Mahalanobis says anomalous
        OR
        directional degradation evidence exceeds
        its NORMAL-calibrated threshold.

    This is deliberately conservative for the first experiment.

    We are NOT claiming this is the final FALCON classifier.
    """

    mahalanobis_anomaly = (
        mahalanobis_score
        > mahalanobis_threshold
    )

    directional_anomaly = (
        directional
        > directional_threshold
    )

    if (
        mahalanobis_anomaly
        or directional_anomaly
    ):
        return "ANOMALOUS"

    return "NORMAL"


# ============================================================
# ONE SEED
# ============================================================

def evaluate_seed(seed):

    calibration_rng = np.random.default_rng(
        seed
    )

    calibration = make_calibration_set(
        n_normal=30,
        time=TIME,
        rng=calibration_rng,
    )

    # --------------------------------------------------------
    # CURRENT MAHALANOBIS BASELINE
    # --------------------------------------------------------

    baseline = build_reference(
        calibration,
        DENSE_GRID,
        gp_kwargs=GP_KWARGS,
    )

    mahalanobis_threshold = calibrate_threshold(
        calibration,
        baseline,
        DENSE_GRID,
        gp_kwargs=GP_KWARGS,
        percentile=MAHALANOBIS_PERCENTILE,
    )

    # --------------------------------------------------------
    # DIRECTIONAL BASELINE
    # --------------------------------------------------------

    directional_reference = (
        build_directional_reference(
            calibration
        )
    )

    directional_threshold = (
        calibrate_directional_threshold(
            calibration,
            directional_reference,
            percentile=MAHALANOBIS_PERCENTILE,
        )
    )

    rows = []

    for tau in TAU_VALUES:

        test_rng = np.random.default_rng(
            seed * 10000
            + int(round(tau * 1000))
        )

        component = make_component(
            component_id=f"TAU-{tau:.2f}",
            time=TIME,
            tau=tau,
            noise_std=NOISE_STD,
            label="TEST",
            rng=test_rng,
        )

        features = extract_features(
            component
        )

        auc = float(
            features[0]
        )

        tau63 = float(
            features[1]
        )

        # ----------------------------------------------------
        # MODEL A: CURRENT MAHALANOBIS
        # ----------------------------------------------------

        mahalanobis = baseline.score(
            features
        )

        decision_mahalanobis = classify(
            mahalanobis,
            mahalanobis_threshold,
        )

        # ----------------------------------------------------
        # MODEL B: DIRECTIONAL ONLY
        # ----------------------------------------------------

        directional, auc_dir, tau63_dir = (
            directional_score(
                features,
                directional_reference,
            )
        )

        decision_directional = (
            "ANOMALOUS"
            if directional
            > directional_threshold
            else "NORMAL"
        )

        # ----------------------------------------------------
        # MODEL C: COMBINED
        # ----------------------------------------------------

        decision_combined = combined_decision(
            mahalanobis,
            mahalanobis_threshold,
            directional,
            directional_threshold,
        )

        rows.append({
            "seed": seed,
            "tau": tau,
            "auc": auc,
            "tau63": tau63,
            "mahalanobis_score": float(
                mahalanobis
            ),
            "mahalanobis_threshold": float(
                mahalanobis_threshold
            ),
            "directional_score": float(
                directional
            ),
            "directional_threshold": float(
                directional_threshold
            ),
            "auc_direction": float(
                auc_dir
            ),
            "tau63_direction": float(
                tau63_dir
            ),
            "decision_mahalanobis":
                decision_mahalanobis,
            "decision_directional":
                decision_directional,
            "decision_combined":
                decision_combined,
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("FALCON CLASSIFIER V2 COMPARISON")
    print("=" * 72)
    print()

    print(
        "Model A: Mahalanobis only"
    )

    print(
        "Model B: Directional score only"
    )

    print(
        "Model C: Mahalanobis OR directional"
    )

    print()

    print(
        f"Seeds       : {len(SEEDS)}"
    )

    print(
        f"Tau values  : {len(TAU_VALUES)}"
    )

    print(
        f"Noise std   : {NOISE_STD}"
    )

    print()

    all_rows = []

    for index, seed in enumerate(SEEDS):

        print(
            f"Running seed "
            f"{index + 1}/{len(SEEDS)}..."
        )

        all_rows.extend(
            evaluate_seed(seed)
        )

    # ========================================================
    # RAW CSV
    # ========================================================

    raw_path = (
        RESULTS_DIR
        / "classifier_v2_raw.csv"
    )

    raw_fields = [
        "seed",
        "tau",
        "auc",
        "tau63",
        "mahalanobis_score",
        "mahalanobis_threshold",
        "directional_score",
        "directional_threshold",
        "auc_direction",
        "tau63_direction",
        "decision_mahalanobis",
        "decision_directional",
        "decision_combined",
    ]

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=raw_fields,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(
        f"Raw results saved: {raw_path}"
    )

    # ========================================================
    # AGGREGATE
    # ========================================================

    summary = []

    for tau in TAU_VALUES:

        rows = [
            row
            for row in all_rows
            if np.isclose(
                row["tau"],
                tau,
            )
        ]

        mahal_rate = np.mean([
            row["decision_mahalanobis"]
            == "ANOMALOUS"
            for row in rows
        ])

        directional_rate = np.mean([
            row["decision_directional"]
            == "ANOMALOUS"
            for row in rows
        ])

        combined_rate = np.mean([
            row["decision_combined"]
            == "ANOMALOUS"
            for row in rows
        ])

        summary.append({
            "tau": tau,

            "mean_auc": float(
                np.mean([
                    row["auc"]
                    for row in rows
                ])
            ),

            "mean_tau63": float(
                np.mean([
                    row["tau63"]
                    for row in rows
                ])
            ),

            "mean_mahalanobis": float(
                np.mean([
                    row["mahalanobis_score"]
                    for row in rows
                ])
            ),

            "mean_directional": float(
                np.mean([
                    row["directional_score"]
                    for row in rows
                ])
            ),

            "mahalanobis_anomaly_rate":
                float(mahal_rate),

            "directional_anomaly_rate":
                float(directional_rate),

            "combined_anomaly_rate":
                float(combined_rate),
        })

    # ========================================================
    # PRINT TABLE
    # ========================================================

    print()
    print("=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print()

    header = (
        f"{'TAU':>6}"
        f"{'MAHAL':>11}"
        f"{'DIR':>11}"
        f"{'COMBINED':>12}"
        f"{'MAHAL %':>11}"
        f"{'DIR %':>10}"
        f"{'COMB %':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in summary:

        print(
            f"{row['tau']:>6.2f}"
            f"{row['mean_mahalanobis']:>11.4f}"
            f"{row['mean_directional']:>11.4f}"
            f"{row['mean_mahalanobis'] > 0:>12}"
            f"{100 * row['mahalanobis_anomaly_rate']:>10.1f}%"
            f"{100 * row['directional_anomaly_rate']:>9.1f}%"
            f"{100 * row['combined_anomaly_rate']:>9.1f}%"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        RESULTS_DIR
        / "classifier_v2_summary.csv"
    )

    summary_fields = list(
        summary[0].keys()
    )

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
        writer.writerows(summary)

    print()
    print(
        f"Summary saved: {summary_path}"
    )

    # ========================================================
    # IMPORTANT COMPARISON:
    # HIGHER TAU = SLOWER THAN NORMAL
    #
    # LOWER TAU = FASTER THAN NORMAL
    #
    # We specifically inspect whether the directional model
    # avoids calling the HIGHER-TAU side anomalous.
    # ========================================================

    high_tau = [
        row
        for row in summary
        if row["tau"] >= 1.04
    ]

    low_tau = [
        row
        for row in summary
        if row["tau"] <= 0.94
    ]

    high_directional_rate = np.mean([
        row["directional_anomaly_rate"]
        for row in high_tau
    ])

    low_directional_rate = np.mean([
        row["directional_anomaly_rate"]
        for row in low_tau
    ])

    print()
    print("=" * 72)
    print("DIRECTIONALITY CHECK")
    print("=" * 72)
    print()

    print(
        "Higher-than-normal tau region "
        "(tau >= 1.04):"
    )

    print(
        f"Directional anomaly rate: "
        f"{100 * high_directional_rate:.1f}%"
    )

    print()

    print(
        "Lower-than-normal tau region "
        "(tau <= 0.94):"
    )

    print(
        f"Directional anomaly rate: "
        f"{100 * low_directional_rate:.1f}%"
    )

    print()

    # ========================================================
    # PLOT 1
    # ========================================================

    taus = np.array([
        row["tau"]
        for row in summary
    ])

    mahal = np.array([
        row["mean_mahalanobis"]
        for row in summary
    ])

    directional = np.array([
        row["mean_directional"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(
        taus,
        mahal,
        marker="o",
        label="Mahalanobis",
    )

    plt.plot(
        taus,
        directional,
        marker="s",
        label="Directional",
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Score")
    plt.title(
        "Mahalanobis vs Directional Evidence"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.gca().invert_xaxis()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / "classifier_v2_scores.png"
    )

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()

    print(
        f"Plot saved: {path}"
    )

    # ========================================================
    # PLOT 2
    # ========================================================

    mahal_rate = np.array([
        row["mahalanobis_anomaly_rate"]
        for row in summary
    ])

    directional_rate = np.array([
        row["directional_anomaly_rate"]
        for row in summary
    ])

    combined_rate = np.array([
        row["combined_anomaly_rate"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(
        taus,
        mahal_rate * 100,
        marker="o",
        label="Mahalanobis",
    )

    plt.plot(
        taus,
        directional_rate * 100,
        marker="s",
        label="Directional",
    )

    plt.plot(
        taus,
        combined_rate * 100,
        marker="^",
        label="Combined",
    )

    plt.xlabel("Simulator tau")
    plt.ylabel(
        "Runs classified ANOMALOUS (%)"
    )

    plt.title(
        "Classifier Decisions Across Tau"
    )

    plt.ylim(
        -5,
        105,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.gca().invert_xaxis()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / "classifier_v2_decision_rates.png"
    )

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()

    print(
        f"Plot saved: {path}"
    )

    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL DIAGNOSTIC CONCLUSION")
    print("=" * 72)
    print()

    if (
        low_directional_rate
        > high_directional_rate
    ):

        print(
            "Directional evidence is more selective for "
            "the lower-tau degradation direction."
        )

        print(
            "This supports investigating a directional "
            "component in the FALCON classifier."
        )

    else:

        print(
            "Directional evidence does not yet provide "
            "the required separation."
        )

        print(
            "Do NOT modify the production classifier."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Model C is intentionally a diagnostic OR-combination."
    )

    print(
        "It is NOT being declared the final FALCON algorithm."
    )

    print(
        "All thresholds are learned from NORMAL calibration "
        "data only."
    )

    print(
        "The tau direction is specific to the synthetic RC "
        "proxy and must not be presented as a universal "
        "semiconductor degradation direction."
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
