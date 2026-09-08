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

# Tau values from healthy -> increasingly fast degradation.
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

# Multiple independent seeds.
SEEDS = list(range(10))

NOISE_STD = 0.015

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
# FIT ONE COMPONENT
# ============================================================

def extract_features(component):
    """
    Fit FALCON's GP trajectory model to the complete trajectory
    and extract AUC + tau63 from the posterior mean.
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
# RUN ONE SEED
# ============================================================

def evaluate_seed(seed):
    """
    Build one NORMAL calibration population and evaluate every
    tau value against that population.
    """

    calibration_rng = np.random.default_rng(seed)

    calibration = make_calibration_set(
        n_normal=30,
        time=TIME,
        rng=calibration_rng,
    )

    baseline = build_reference(
        calibration,
        DENSE_GRID,
        gp_kwargs=GP_KWARGS,
    )

    threshold = calibrate_threshold(
        calibration,
        baseline,
        DENSE_GRID,
        gp_kwargs=GP_KWARGS,
        percentile=THRESHOLD_PERCENTILE,
    )

    rows = []

    for tau in TAU_VALUES:

        # Independent noise RNG for each tau.
        test_rng = np.random.default_rng(
            seed * 10000 + int(round(tau * 1000))
        )

        component = make_component(
            component_id=f"TAU-{tau:.2f}",
            time=TIME,
            tau=tau,
            noise_std=NOISE_STD,
            label="TEST",
            rng=test_rng,
        )

        features = extract_features(component)

        score = baseline.score(features)

        decision = classify(
            score,
            threshold,
        )

        rows.append({
            "seed": seed,
            "tau": tau,
            "auc": float(features[0]),
            "tau63": float(features[1]),
            "score": float(score),
            "threshold": float(threshold),
            "decision": decision,
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("FALCON TAU BOUNDARY EVALUATION")
    print("=" * 72)
    print()

    print(f"Seeds          : {len(SEEDS)}")
    print(f"Tau values     : {len(TAU_VALUES)}")
    print(f"Noise std      : {NOISE_STD}")
    print(
        f"Threshold rule : "
        f"{THRESHOLD_PERCENTILE:.0f}th percentile of NORMAL scores"
    )
    print()

    all_rows = []

    for seed in SEEDS:

        print(
            f"Running seed {seed + 1}/{len(SEEDS)}..."
        )

        rows = evaluate_seed(seed)

        all_rows.extend(rows)

    # ========================================================
    # SAVE RAW CSV
    # ========================================================

    raw_path = RESULTS_DIR / "tau_boundary_raw.csv"

    fieldnames = [
        "seed",
        "tau",
        "auc",
        "tau63",
        "score",
        "threshold",
        "decision",
    ]

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(all_rows)

    print()
    print(f"Raw results saved: {raw_path}")

    # ========================================================
    # AGGREGATE BY TAU
    # ========================================================

    summary = []

    for tau in TAU_VALUES:

        rows = [
            row
            for row in all_rows
            if np.isclose(row["tau"], tau)
        ]

        scores = np.array([
            row["score"]
            for row in rows
        ])

        aucs = np.array([
            row["auc"]
            for row in rows
        ])

        tau63s = np.array([
            row["tau63"]
            for row in rows
        ])

        thresholds = np.array([
            row["threshold"]
            for row in rows
        ])

        anomalous_count = sum(
            row["decision"] == "ANOMALOUS"
            for row in rows
        )

        anomaly_rate = (
            anomalous_count / len(rows)
        )

        summary.append({
            "tau": tau,
            "mean_auc": float(aucs.mean()),
            "std_auc": float(aucs.std(ddof=1)),
            "mean_tau63": float(tau63s.mean()),
            "std_tau63": float(tau63s.std(ddof=1)),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=1)),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "mean_threshold": float(thresholds.mean()),
            "anomaly_rate": float(anomaly_rate),
        })

    # ========================================================
    # PRINT TABLE
    # ========================================================

    print()
    print("=" * 72)
    print("TAU BOUNDARY SUMMARY")
    print("=" * 72)

    print()

    header = (
        f"{'TAU':>6}"
        f"{'AUC':>12}"
        f"{'TAU63':>12}"
        f"{'SCORE':>12}"
        f"{'SCORE SD':>12}"
        f"{'ANOM %':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in summary:

        print(
            f"{row['tau']:>6.2f}"
            f"{row['mean_auc']:>12.5f}"
            f"{row['mean_tau63']:>12.5f}"
            f"{row['mean_score']:>12.5f}"
            f"{row['std_score']:>12.5f}"
            f"{100 * row['anomaly_rate']:>9.1f}%"
        )

    # ========================================================
    # DETERMINE EMPIRICAL TRANSITION
    # ========================================================

    print()
    print("=" * 72)
    print("EMPIRICAL DECISION TRANSITION")
    print("=" * 72)
    print()

    # Sort from slowest/healthiest to fastest.
    ordered = sorted(
        summary,
        key=lambda x: x["tau"],
        reverse=True,
    )

    # Find the first tau where >=95% of runs are anomalous.
    robust_anomaly_tau = None

    for row in ordered:

        if row["anomaly_rate"] >= 0.95:

            robust_anomaly_tau = row["tau"]

    # Find the smallest tau where <=5% are anomalous.
    robust_normal_tau = None

    for row in ordered:

        if row["anomaly_rate"] <= 0.05:

            robust_normal_tau = row["tau"]

    print(
        "Interpretation is empirical only."
    )

    print()

    if robust_normal_tau is not None:

        print(
            f"Highest tested tau with <=5% anomalous runs: "
            f"{robust_normal_tau:.2f}"
        )

    else:

        print(
            "No tau value had <=5% anomalous runs."
        )

    if robust_anomaly_tau is not None:

        print(
            f"Lowest tested tau reaching >=95% anomalous runs: "
            f"{robust_anomaly_tau:.2f}"
        )

    else:

        print(
            "No tau value reached >=95% anomalous runs."
        )

    # ========================================================
    # FIND UNCERTAIN TRANSITION BAND
    # ========================================================

    transition_rows = [
        row
        for row in summary
        if 0.05 < row["anomaly_rate"] < 0.95
    ]

    print()

    if transition_rows:

        transition_taus = [
            row["tau"]
            for row in transition_rows
        ]

        print(
            "Transition band observed at tau values:"
        )

        print(
            "  "
            + ", ".join(
                f"{tau:.2f}"
                for tau in transition_taus
            )
        )

    else:

        print(
            "No intermediate transition band observed "
            "at the tested tau resolution."
        )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary_path = RESULTS_DIR / "tau_boundary_summary.csv"

    summary_fields = [
        "tau",
        "mean_auc",
        "std_auc",
        "mean_tau63",
        "std_tau63",
        "mean_score",
        "std_score",
        "min_score",
        "max_score",
        "mean_threshold",
        "anomaly_rate",
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

        writer.writerows(summary)

    print()
    print(f"Summary saved: {summary_path}")

    # ========================================================
    # PLOT 1: SCORE VS TAU
    # ========================================================

    taus = np.array([
        row["tau"]
        for row in summary
    ])

    mean_scores = np.array([
        row["mean_score"]
        for row in summary
    ])

    std_scores = np.array([
        row["std_score"]
        for row in summary
    ])

    mean_thresholds = np.array([
        row["mean_threshold"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        taus,
        mean_scores,
        yerr=std_scores,
        marker="o",
        capsize=4,
        label="Mean Mahalanobis score ± 1 SD",
    )

    plt.plot(
        taus,
        mean_thresholds,
        linestyle="--",
        label="Mean anomaly threshold",
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Mahalanobis score")
    plt.title("FALCON Anomaly Score vs Tau")
    plt.grid(True, alpha=0.25)
    plt.legend()

    # Larger tau = slower/healthier, so show healthy side first.
    plt.gca().invert_xaxis()

    plt.tight_layout()

    score_plot = RESULTS_DIR / "tau_boundary_score.png"

    plt.savefig(
        score_plot,
        dpi=180,
    )

    plt.close()

    print(f"Plot saved: {score_plot}")

    # ========================================================
    # PLOT 2: ANOMALY RATE VS TAU
    # ========================================================

    anomaly_rates = np.array([
        row["anomaly_rate"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(
        taus,
        anomaly_rates * 100.0,
        marker="o",
    )

    plt.axhline(
        5.0,
        linestyle="--",
        label="5% anomalous",
    )

    plt.axhline(
        95.0,
        linestyle="--",
        label="95% anomalous",
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Runs classified ANOMALOUS (%)")
    plt.title("FALCON Decision Transition vs Tau")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.25)
    plt.legend()

    plt.gca().invert_xaxis()

    plt.tight_layout()

    rate_plot = RESULTS_DIR / "tau_boundary_anomaly_rate.png"

    plt.savefig(
        rate_plot,
        dpi=180,
    )

    plt.close()

    print(f"Plot saved: {rate_plot}")

    # ========================================================
    # PLOT 3: AUC VS TAU
    # ========================================================

    mean_auc = np.array([
        row["mean_auc"]
        for row in summary
    ])

    std_auc = np.array([
        row["std_auc"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        taus,
        mean_auc,
        yerr=std_auc,
        marker="o",
        capsize=4,
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("AUC")
    plt.title("Trajectory AUC vs Tau")
    plt.grid(True, alpha=0.25)

    plt.gca().invert_xaxis()

    plt.tight_layout()

    auc_plot = RESULTS_DIR / "tau_boundary_auc.png"

    plt.savefig(
        auc_plot,
        dpi=180,
    )

    plt.close()

    print(f"Plot saved: {auc_plot}")

    # ========================================================
    # PLOT 4: TAU63 VS TAU
    # ========================================================

    mean_tau63 = np.array([
        row["mean_tau63"]
        for row in summary
    ])

    std_tau63 = np.array([
        row["std_tau63"]
        for row in summary
    ])

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        taus,
        mean_tau63,
        yerr=std_tau63,
        marker="o",
        capsize=4,
    )

    plt.xlabel("Simulator tau")
    plt.ylabel("Extracted tau63")
    plt.title("Extracted tau63 vs Simulator Tau")
    plt.grid(True, alpha=0.25)

    plt.gca().invert_xaxis()

    plt.tight_layout()

    tau63_plot = RESULTS_DIR / "tau_boundary_tau63.png"

    plt.savefig(
        tau63_plot,
        dpi=180,
    )

    plt.close()

    print(f"Plot saved: {tau63_plot}")

    # ========================================================
    # FINAL WARNING
    # ========================================================

    print()
    print("=" * 72)
    print("IMPORTANT")
    print("=" * 72)
    print()

    print(
        "This experiment measures the behaviour of the CURRENT "
        "synthetic FALCON classifier."
    )

    print(
        "It does NOT establish a physical semiconductor failure "
        "threshold."
    )

    print(
        "Do not call the empirical tau boundary a real-world "
        "device-failure boundary."
    )

    print()
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
