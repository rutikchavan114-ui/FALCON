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

SEED = 42

# Same time grid used by the current FALCON simulator.
TIME = np.linspace(0.0, 5.0, 101)

# Dense grid used for GP feature extraction.
DENSE_GRID = np.linspace(0.0, 5.0, 501)

# Keep this identical to the current MVP GP configuration.
GP_KWARGS = {
    "length_scale": 0.75,
    "signal_variance": 4.0,
    "noise_variance": 0.015 ** 2,
    "random_state": 0,
}

# Existing anomaly threshold definition.
THRESHOLD_PERCENTILE = 95.0

# Output directory.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fit_and_extract_features(component):
    """
    Fit the same GP used by FALCON and extract:

        feature[0] = AUC
        feature[1] = tau63

    The features are calculated from the GP posterior mean,
    matching the existing build_reference() / calibrate_threshold()
    implementation.
    """

    gp = ComponentGP(**GP_KWARGS)

    gp.fit(
        component.time,
        component.full_observation
    )

    mean, _ = gp.predict(DENSE_GRID)

    features = trajectory_features(
        DENSE_GRID,
        mean
    )

    return gp, mean, features


def make_diagnostic_test_set():
    """
    Construct diagnostic cases spanning the current simulator's
    normal -> borderline -> anomalous region.

    IMPORTANT:
    These are synthetic RC trajectories used only to inspect the
    behaviour of the current MVP classifier.

    They are NOT semiconductor degradation data.
    """

    rng = np.random.default_rng(SEED)

    components = []

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    normal_cases = [
        ("N01", 1.00, "NORMAL"),
        ("N02", 1.03, "NORMAL"),
    ]

    # --------------------------------------------------------
    # BORDERLINE
    #
    # Current robustness experiment uses 0.90 and 0.85.
    # They are intentionally labelled NORMAL in the experiment,
    # so we can see why the current classifier rejects them.
    # --------------------------------------------------------

    borderline_cases = [
        ("B01", 0.90, "BORDERLINE"),
        ("B02", 0.85, "BORDERLINE"),
    ]

    # --------------------------------------------------------
    # ANOMALOUS
    # --------------------------------------------------------

    anomalous_cases = [
        ("A01", 0.70, "ANOMALOUS"),
        ("A02", 0.60, "ANOMALOUS"),
    ]

    all_cases = (
        normal_cases
        + borderline_cases
        + anomalous_cases
    )

    for component_id, tau, label in all_cases:

        component = make_component(
            component_id=component_id,
            time=TIME,
            tau=tau,
            noise_std=0.015,
            label=label,
            rng=rng,
        )

        components.append(component)

    return components


def calculate_mahalanobis_components(features, baseline):
    """
    Return the two-dimensional Mahalanobis contribution.

    This is diagnostic only.

    The actual FALCON score remains:

        d.T @ precision @ d
    """

    d = np.asarray(features, dtype=float) - baseline.mean

    # Whiten the deviation using the precision matrix.
    # This lets us inspect the relative contribution of
    # each feature in the covariance-aware feature space.
    try:
        eigvals, eigvecs = np.linalg.eigh(baseline.precision)

        eigvals = np.maximum(eigvals, 0.0)

        transformed = (
            eigvecs
            @ np.diag(np.sqrt(eigvals))
            @ eigvecs.T
            @ d
        )

        contributions = transformed ** 2

    except np.linalg.LinAlgError:
        contributions = np.array([np.nan, np.nan])

    return contributions


# ============================================================
# BUILD CALIBRATION BASELINE
# ============================================================

print()
print("=" * 72)
print("FALCON FEATURE-SPACE DIAGNOSTIC")
print("=" * 72)
print()

print("Creating NORMAL calibration set...")

calibration_components = make_calibration_set(
    n_normal=30,
    time=TIME,
    rng=np.random.default_rng(7),
)

print(f"Calibration components : {len(calibration_components)}")
print()


# ============================================================
# BUILD THE SAME REFERENCE BASELINE AS FALCON
# ============================================================

print("Building ReferenceBaseline...")

baseline = build_reference(
    calibration_components,
    DENSE_GRID,
    gp_kwargs=GP_KWARGS,
)

threshold = calibrate_threshold(
    calibration_components,
    baseline,
    DENSE_GRID,
    gp_kwargs=GP_KWARGS,
    percentile=THRESHOLD_PERCENTILE,
)

print(f"Anomaly threshold ({THRESHOLD_PERCENTILE:.0f}th percentile): "
      f"{threshold:.6f}")

print()
print("Reference feature mean:")
print(f"  AUC   = {baseline.mean[0]:.6f}")
print(f"  tau63 = {baseline.mean[1]:.6f}")
print()


# ============================================================
# INSPECT CALIBRATION DISTRIBUTION
# ============================================================

calibration_rows = []

for component in calibration_components:

    _, _, features = fit_and_extract_features(component)

    score = baseline.score(features)

    calibration_rows.append({
        "component_id": component.component_id,
        "group": "CALIBRATION",
        "label": component.label,
        "tau_simulator": component.tau,
        "auc": features[0],
        "tau63": features[1],
        "mahalanobis_score": score,
        "decision": classify(score, threshold),
    })


# ============================================================
# CREATE TEST CASES
# ============================================================

print("Creating diagnostic test cases...")

test_components = make_diagnostic_test_set()

print(f"Test components         : {len(test_components)}")
print()


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

test_rows = []

for component in test_components:

    _, _, features = fit_and_extract_features(component)

    score = baseline.score(features)

    contributions = calculate_mahalanobis_components(
        features,
        baseline
    )

    decision = classify(
        score,
        threshold
    )

    test_rows.append({
        "component_id": component.component_id,
        "group": component.label,
        "label": component.label,
        "tau_simulator": component.tau,
        "auc": features[0],
        "tau63": features[1],
        "mahalanobis_score": score,
        "decision": decision,
        "auc_contribution": contributions[0],
        "tau63_contribution": contributions[1],
    })


# ============================================================
# PRINT CALIBRATION SUMMARY
# ============================================================

calibration_scores = np.array([
    row["mahalanobis_score"]
    for row in calibration_rows
])

print("=" * 72)
print("CALIBRATION SUMMARY")
print("=" * 72)

print(f"Minimum calibration score : {calibration_scores.min():.6f}")
print(f"Median calibration score  : {np.median(calibration_scores):.6f}")
print(f"Mean calibration score    : {calibration_scores.mean():.6f}")
print(f"95th percentile           : {np.percentile(calibration_scores, 95):.6f}")
print(f"Maximum calibration score : {calibration_scores.max():.6f}")

print()


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print("=" * 72)
print("TEST FEATURE-SPACE RESULTS")
print("=" * 72)

header = (
    f"{'ID':<6}"
    f"{'GROUP':<13}"
    f"{'TAU':>8}"
    f"{'AUC':>14}"
    f"{'TAU63':>14}"
    f"{'SCORE':>14}"
    f"{'DECISION':>12}"
)

print(header)
print("-" * len(header))

for row in test_rows:

    print(
        f"{row['component_id']:<6}"
        f"{row['group']:<13}"
        f"{row['tau_simulator']:>8.3f}"
        f"{row['auc']:>14.6f}"
        f"{row['tau63']:>14.6f}"
        f"{row['mahalanobis_score']:>14.6f}"
        f"{row['decision']:>12}"
    )

print()


# ============================================================
# FEATURE-SPACE INTERPRETATION
# ============================================================

print("=" * 72)
print("FEATURE-SPACE INTERPRETATION")
print("=" * 72)

print()
print("Reference mean:")
print(f"  AUC   = {baseline.mean[0]:.6f}")
print(f"  tau63 = {baseline.mean[1]:.6f}")
print()

print("Threshold:")
print(f"  Mahalanobis score > {threshold:.6f} => ANOMALOUS")
print()

for row in test_rows:

    delta_auc = row["auc"] - baseline.mean[0]
    delta_tau = row["tau63"] - baseline.mean[1]

    print(
        f"{row['component_id']}: "
        f"AUC deviation={delta_auc:+.6f}, "
        f"tau63 deviation={delta_tau:+.6f}, "
        f"score={row['mahalanobis_score']:.6f}, "
        f"decision={row['decision']}"
    )

print()


# ============================================================
# GROUP STATISTICS
# ============================================================

print("=" * 72)
print("GROUP STATISTICS")
print("=" * 72)

groups = [
    "NORMAL",
    "BORDERLINE",
    "ANOMALOUS",
]

for group in groups:

    rows = [
        row
        for row in test_rows
        if row["group"] == group
    ]

    if not rows:
        continue

    auc_values = np.array([
        row["auc"]
        for row in rows
    ])

    tau_values = np.array([
        row["tau63"]
        for row in rows
    ])

    score_values = np.array([
        row["mahalanobis_score"]
        for row in rows
    ])

    decisions = [
        row["decision"]
        for row in rows
    ]

    print()
    print(group)

    print(
        f"  AUC    : "
        f"mean={auc_values.mean():.6f}, "
        f"range=[{auc_values.min():.6f}, {auc_values.max():.6f}]"
    )

    print(
        f"  tau63  : "
        f"mean={tau_values.mean():.6f}, "
        f"range=[{tau_values.min():.6f}, {tau_values.max():.6f}]"
    )

    print(
        f"  Score  : "
        f"mean={score_values.mean():.6f}, "
        f"range=[{score_values.min():.6f}, {score_values.max():.6f}]"
    )

    print(
        f"  Decisions: {decisions}"
    )


print()


# ============================================================
# SAVE CSV
# ============================================================

csv_path = RESULTS_DIR / "feature_space_diagnostic.csv"

fieldnames = [
    "component_id",
    "group",
    "label",
    "tau_simulator",
    "auc",
    "tau63",
    "mahalanobis_score",
    "decision",
    "auc_contribution",
    "tau63_contribution",
]

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in test_rows:
        writer.writerow(row)

print(f"Saved CSV: {csv_path}")


# ============================================================
# PLOT 1:
# AUC vs TAU63 FEATURE SPACE
# ============================================================

plt.figure(figsize=(9, 7))

# Calibration NORMAL points.
cal_auc = [
    row["auc"]
    for row in calibration_rows
]

cal_tau = [
    row["tau63"]
    for row in calibration_rows
]

plt.scatter(
    cal_auc,
    cal_tau,
    marker="o",
    alpha=0.45,
    label="Calibration NORMAL",
)

# Test groups.
markers = {
    "NORMAL": "o",
    "BORDERLINE": "^",
    "ANOMALOUS": "x",
}

for group in groups:

    rows = [
        row
        for row in test_rows
        if row["group"] == group
    ]

    if not rows:
        continue

    plt.scatter(
        [row["auc"] for row in rows],
        [row["tau63"] for row in rows],
        marker=markers[group],
        s=100,
        label=f"Test {group}",
    )

# Reference mean.
plt.scatter(
    [baseline.mean[0]],
    [baseline.mean[1]],
    marker="*",
    s=180,
    label="Reference mean",
)

# Label test points.
for row in test_rows:

    plt.annotate(
        row["component_id"],
        (
            row["auc"],
            row["tau63"],
        ),
        xytext=(6, 6),
        textcoords="offset points",
    )

plt.xlabel("AUC")
plt.ylabel("tau63")
plt.title("FALCON Feature Space: AUC vs tau63")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

feature_plot_path = RESULTS_DIR / "feature_space_auc_tau63.png"

plt.savefig(
    feature_plot_path,
    dpi=180,
)

plt.close()

print(f"Saved plot: {feature_plot_path}")


# ============================================================
# PLOT 2:
# MAHALANOBIS SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    calibration_scores,
    bins=12,
    alpha=0.55,
    label="Calibration NORMAL",
)

for group in groups:

    scores = [
        row["mahalanobis_score"]
        for row in test_rows
        if row["group"] == group
    ]

    if not scores:
        continue

    plt.scatter(
        scores,
        np.zeros(len(scores)),
        s=90,
        marker=markers[group],
        label=f"Test {group}",
    )

plt.axvline(
    threshold,
    linestyle="--",
    label=f"95th percentile threshold = {threshold:.4f}",
)

plt.xlabel("Mahalanobis score")
plt.ylabel("Frequency")
plt.title("FALCON Mahalanobis Score Distribution")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

score_plot_path = RESULTS_DIR / "feature_space_mahalanobis.png"

plt.savefig(
    score_plot_path,
    dpi=180,
)

plt.close()

print(f"Saved plot: {score_plot_path}")


# ============================================================
# FINAL DIAGNOSTIC CONCLUSION
# ============================================================

print()
print("=" * 72)
print("DIAGNOSTIC CONCLUSION")
print("=" * 72)

borderline_rows = [
    row
    for row in test_rows
    if row["group"] == "BORDERLINE"
]

borderline_anomalous = sum(
    row["decision"] == "ANOMALOUS"
    for row in borderline_rows
)

print()

if borderline_anomalous == len(borderline_rows):

    print(
        "Both borderline cases are classified as ANOMALOUS."
    )

    print(
        "This indicates that the current 2-feature Mahalanobis "
        "classifier does not separate the chosen borderline "
        "region from the anomalous region."
    )

elif borderline_anomalous > 0:

    print(
        "The borderline region crosses the current anomaly boundary."
    )

    print(
        "This suggests the current classifier has a transition "
        "region rather than a clean separation."
    )

else:

    print(
        "Both borderline cases remain NORMAL under the current "
        "95th-percentile Mahalanobis threshold."
    )

print()

print(
    "IMPORTANT: Do NOT change the classifier yet based only on this "
    "script."
)

print(
    "Use the AUC/tau63 feature-space plot and score distribution "
    "to determine whether the problem is feature overlap, "
    "covariance/thresholding, or the synthetic simulator itself."
)

print()

print("=" * 72)
print("DONE")
print("=" * 72)
