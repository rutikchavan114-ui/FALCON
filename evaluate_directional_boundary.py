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

# Same tau sweep used in the previous experiment.
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

THRESHOLD_PERCENTILE = 95.0

# Directional rule:
#
# For the current RC proxy:
# lower AUC  = faster response
# lower tau63 = faster response
#
# Therefore lower-than-normal values are treated as
# degradation-direction evidence.
#
# This is ONLY valid for the current RC simulator.
DIRECTION = "LOWER_IS_DEGRADATION"

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
    Fit the current FALCON GP to the complete trajectory and
    extract [AUC, tau63] from the posterior mean.
    """

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
# ROBUST NORMAL DIRECTIONAL SCALE
# ============================================================

def robust_scale(values):
    """
    Robust scale based on MAD.

    scale = 1.4826 * median(|x - median(x)|)

    This prevents one unusual calibration sample from
    dominating the directional standardized score.
    """

    values = np.asarray(values, dtype=float)

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    scale = 1.4826 * mad

    # Prevent division by zero for an artificially perfect
    # synthetic calibration population.
    if scale < 1e-9:
        scale = np.std(values)

    if scale < 1e-9:
        scale = 1e-9

    return float(scale)


# ============================================================
# BUILD DIRECTIONAL REFERENCE
# ============================================================

def build_directional_reference(calibration):

    feature_matrix = []

    for component in calibration:

        feature_matrix.append(
            extract_features(component)
        )

    feature_matrix = np.asarray(
        feature_matrix,
        dtype=float,
    )

    median = np.median(
        feature_matrix,
        axis=0,
    )

    scales = np.array([
        robust_scale(feature_matrix[:, 0]),
        robust_scale(feature_matrix[:, 1]),
    ])

    mean = np.mean(
        feature_matrix,
        axis=0,
    )

    return {
        "features": feature_matrix,
        "median": median,
        "mean": mean,
        "scales": scales,
    }


# ============================================================
# DIRECTIONAL SCORE
# ============================================================

def directional_score(features, reference):
    """
    Calculate degradation-direction evidence.

    For the current RC proxy:

        lower AUC   -> degradation evidence
        lower tau63 -> degradation evidence

    Positive values mean movement toward the degradation
    direction.

    Each feature is robustly standardized before combining.

    We use the positive part only:

        max(0, reference - observed) / scale

    This means slower-than-normal behaviour does not become
    degradation evidence.
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

    # Equal weighting for the current two-feature proxy.
    score = np.sqrt(
        auc_positive ** 2
        +
        tau_positive ** 2
    )

    return float(score), float(auc_direction), float(tau_direction)


# ============================================================
# RUN ONE SEED
# ============================================================

def evaluate_seed(seed):

    calibration_rng = np.random.default_rng(seed)

    calibration = make_calibration_set(
        n_normal=30,
        time=TIME,
        rng=calibration_rng,
    )

    # Existing FALCON statistical baseline.
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
        percentile=THRESHOLD_PERCENTILE,
    )

    # New diagnostic directional baseline.
    directional_reference = build_directional_reference(
        calibration
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

        # Existing detector.
        mahalanobis = baseline.score(
            features
        )

        mahalanobis_decision = classify(
            mahalanobis,
            mahalanobis_threshold,
        )

        # New directional diagnostic.
        directional, auc_dir, tau63_dir = directional_score(
            features,
            directional_reference,
        )

        rows.append({
            "seed": seed,
            "tau": tau,
            "auc": float(features[0]),
            "tau63": float(features[1]),
            "mahalanobis_score": float(mahalanobis),
            "mahalanobis_threshold": float(
                mahalanobis_threshold
            ),
            "mahalanobis_decision": mahalanobis_decision,
            "directional_score": float(directional),
            "auc_direction": float(auc_dir),
            "tau63_direction": float(tau63_dir),
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("FALCON DIRECTIONAL BOUNDARY EVALUATION")
    print("=" * 72)
    print()

    print(f"Seeds              : {len(SEEDS)}")
    print(f"Tau values         : {len(TAU_VALUES)}")
    print(f"Noise std          : {NOISE_STD}")
    print(f"Features            : AUC + tau63")
    print(f"Direction           : {DIRECTION}")
    print()

    all_rows = []

    for index, seed in enumerate(SEEDS):

        print(
            f"Running seed {index + 1}/{len(SEEDS)}..."
        )

        all_rows.extend(
            evaluate_seed(seed)
        )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_path = (
        RESULTS_DIR
        / "directional_boundary_raw.csv"
    )

    raw_fields = [
        "seed",
        "tau",
        "auc",
        "tau63",
        "mahalanobis_score",
        "mahalanobis_threshold",
        "mahalanobis_decision",
        "directional_score",
        "auc_direction",
        "tau63_direction",
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
    print(f"Raw results saved: {raw_path}")

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

        auc = np.array([
            row["auc"]
            for row in rows
        ])

        tau63 = np.array([
            row["tau63"]
            for row in rows
        ])

        mahal = np.array([
            row["mahalanobis_score"]
            for row in rows
        ])

        directional = np.array([
            row["directional_score"]
            for row in rows
        ])

        auc_direction = np.array([
            row["auc_direction"]
            for row in rows
        ])

        tau63_direction = np.array([
            row["tau63_direction"]
            for row in rows
        ])

        anomaly_rate = np.mean([
            row["mahalanobis_decision"]
            == "ANOMALOUS"
            for row in rows
        ])

        summary.append({
            "tau": tau,
            "mean_auc": float(auc.mean()),
            "mean_tau63": float(tau63.mean()),
            "mean_mahalanobis": float(mahal.mean()),
            "mean_directional": float(
                directional.mean()
            ),
            "std_directional": float(
                directional.std(ddof=1)
            ),
            "mean_auc_direction": float(
                auc_direction.mean()
            ),
            "mean_tau63_direction": float(
                tau63_direction.mean()
            ),
            "mahalanobis_anomaly_rate": float(
                anomaly_rate
            ),
        })

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("DIRECTIONAL BOUNDARY SUMMARY")
    print("=" * 72)
    print()

    header = (
        f"{'TAU':>6}"
        f"{'AUC':>11}"
        f"{'TAU63':>11}"
        f"{'MAHAL':>12}"
        f"{'DIR SCORE':>12}"
        f"{'AUC DIR':>11}"
        f"{'TAU63 DIR':>12}"
        f"{'ANOM %':>9}"
    )

    print(header)
    print("-" * len(header))

    for row in summary:

        print(
            f"{row['tau']:>6.2f}"
            f"{row['mean_auc']:>11.5f}"
            f"{row['mean_tau63']:>11.5f}"
            f"{row['mean_mahalanobis']:>12.5f}"
            f"{row['mean_directional']:>12.5f}"
            f"{row['mean_auc_direction']:>11.3f}"
            f"{row['mean_tau63_direction']:>12.3f}"
            f"{100 * row['mahalanobis_anomaly_rate']:>8.1f}%"
        )

    # ========================================================
    # FIND DIRECTIONAL CROSSOVER
    # ========================================================

    print()
    print("=" * 72)
    print("DIRECTIONAL INTERPRETATION")
    print("=" * 72)
    print()

    print(
        "Directional score > 0 means the trajectory moved "
        "below the NORMAL reference in at least one feature."
    )

    print(
        "Higher directional score means stronger movement "
        "in the current simulator's degradation direction."
    )

    print()

    # ========================================================
    # CHECK MONOTONICITY
    # ========================================================

    # For degradation direction, smaller tau should generally
    # produce larger directional evidence.
    #
    # We measure Spearman correlation without scipy.
    #
    # Rank-based correlation is calculated manually.

    tau_array = np.array([
        row["tau"]
        for row in summary
    ])

    direction_array = np.array([
        row["mean_directional"]
        for row in summary
    ])

    def ranks(values):

        order = np.argsort(values)

        ranks = np.empty(
            len(values),
            dtype=float,
        )

        ranks[order] = np.arange(
            len(values),
            dtype=float,
        )

        return ranks

    tau_ranks = ranks(
        tau_array
    )

    direction_ranks = ranks(
        direction_array
    )

    tau_centered = (
        tau_ranks
        - tau_ranks.mean()
    )

    direction_centered = (
        direction_ranks
        - direction_ranks.mean()
    )

    denominator = (
        np.sqrt(
            np.sum(tau_centered ** 2)
        )
        *
        np.sqrt(
            np.sum(direction_centered ** 2)
        )
    )

    if denominator > 0:

        spearman = (
            np.sum(
                tau_centered
                * direction_centered
            )
            / denominator
        )

    else:

        spearman = np.nan

    print(
        f"Spearman correlation "
        f"(tau vs directional score): "
        f"{spearman:.4f}"
    )

    if np.isfinite(spearman):

        if spearman < -0.8:

            print(
                "Strong monotonic relationship detected: "
                "lower tau generally produces stronger "
                "degradation-direction evidence."
            )

        elif spearman < -0.5:

            print(
                "Moderate monotonic relationship detected."
            )

        else:

            print(
                "Weak/non-monotonic relationship detected. "
                "Do NOT use directional scoring as a final "
                "classifier yet."
            )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary_path = (
        RESULTS_DIR
        / "directional_boundary_summary.csv"
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
    # PLOT 1: DIRECTIONAL SCORE VS TAU
    # ========================================================

    taus = np.array([
        row["tau"]
        for row in summary
    ])

    mean_directional = np.array([
        row["mean_directional"]
        for row in summary
    ])

    std_directional = np.array([
        row["std_directional"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        taus,
        mean_directional,
        yerr=std_directional,
        marker="o",
        capsize=4,
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Directional degradation score")
    plt.title(
        "Directional Degradation Evidence vs Tau"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.gca().invert_xaxis()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / "directional_score_vs_tau.png"
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
    # PLOT 2: MAHALANOBIS VS DIRECTIONAL SCORE
    # ========================================================

    mean_mahal = np.array([
        row["mean_mahalanobis"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(
        taus,
        mean_mahal,
        marker="o",
        label="Mahalanobis score",
    )

    plt.plot(
        taus,
        mean_directional,
        marker="s",
        label="Directional score",
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Score")
    plt.title(
        "General Anomaly vs Directional Evidence"
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
        / "mahalanobis_vs_directional.png"
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
    # PLOT 3: FEATURE DIRECTIONS
    # ========================================================

    mean_auc_direction = np.array([
        row["mean_auc_direction"]
        for row in summary
    ])

    mean_tau63_direction = np.array([
        row["mean_tau63_direction"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(
        taus,
        mean_auc_direction,
        marker="o",
        label="AUC directional evidence",
    )

    plt.plot(
        taus,
        mean_tau63_direction,
        marker="s",
        label="tau63 directional evidence",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Standardized direction")
    plt.title(
        "Feature Direction Relative to Normal Reference"
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
        / "feature_direction_vs_tau.png"
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
    print("CONCLUSION")
    print("=" * 72)
    print()

    if (
        np.isfinite(spearman)
        and spearman < -0.8
    ):

        print(
            "The current RC simulator shows a strong "
            "directional relationship."
        )

        print(
            "Lower tau consistently corresponds to stronger "
            "degradation-direction evidence."
        )

        print()
        print(
            "This supports testing a directional component "
            "inside FALCON."
        )

    else:

        print(
            "The current evidence does not justify replacing "
            "the existing anomaly detector with directional "
            "scoring."
        )

        print()
        print(
            "Keep the current classifier unchanged."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a diagnostic experiment on the synthetic "
        "RC proxy. It is NOT a semiconductor degradation "
        "model and does not establish a physical failure "
        "criterion."
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
