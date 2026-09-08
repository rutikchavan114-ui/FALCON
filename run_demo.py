from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from falcon.simulator import make_calibration_set, make_test_set
from falcon.pipeline import (
    build_reference,
    calibrate_threshold,
    fixed_rate_classification,
    adaptive_classification,
)
from falcon.gp_model import ComponentGP


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    dense_grid = np.linspace(0.0, 5.0, 201)
    physical_grid = np.linspace(0.0, 5.0, 101)

    calibration = make_calibration_set(
        n_normal=30, time=physical_grid, rng=np.random.default_rng(123)
    )
    test = make_test_set(
        n_normal=4, n_anomaly=1, time=physical_grid,
        rng=np.random.default_rng(456)
    )

    gp_kwargs = {
        "length_scale": 0.75,
        "signal_variance": 4.0,
        "noise_variance": 0.015**2,
    }

    baseline = build_reference(calibration, dense_grid, gp_kwargs)
    threshold = calibrate_threshold(
        calibration, baseline, dense_grid, gp_kwargs, percentile=95.0
    )

    fixed_indices = np.arange(0, len(physical_grid), 1)
    rows = []

    for comp in test:
        fixed = fixed_rate_classification(
            comp, fixed_indices, dense_grid, baseline, threshold, gp_kwargs
        )
        adaptive = adaptive_classification(
            comp, dense_grid, baseline, threshold,
            base_interval=0.05, max_interval=0.50,
            p_threshold=0.05, n_samples=50,
            gp_kwargs=gp_kwargs, seed=100
        )
        rows.append({
            "component": comp.component_id,
            "ground_truth": comp.label,
            "fixed_decision": fixed["decision"],
            "falcon_decision": adaptive["decision"],
            "fixed_measurements": fixed["measurement_count"],
            "falcon_measurements": adaptive["measurement_count"],
            "fixed_score": fixed["score"],
            "falcon_score": adaptive["score"],
            "classification_agreement":
                fixed["decision"] == adaptive["decision"],
        })

    fixed_total = sum(r["fixed_measurements"] for r in rows)
    falcon_total = sum(r["falcon_measurements"] for r in rows)
    reduction = 1.0 - falcon_total / fixed_total
    agreement = float(np.mean(
        [r["classification_agreement"] for r in rows]
    ))

    result = {
        "note": "Simulation result only; not an industrial performance claim.",
        "threshold": threshold,
        "components": rows,
        "total_fixed_measurements": fixed_total,
        "total_falcon_measurements": falcon_total,
        "simulated_measurement_reduction": reduction,
        "classification_agreement": agreement,
    }
    (RESULTS / "demo_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    # Trajectory plot for the anomalous example.
    example = test[-1]
    adaptive_example = adaptive_classification(
        example, dense_grid, baseline, threshold,
        base_interval=0.05, max_interval=0.50,
        p_threshold=0.05, n_samples=50,
        gp_kwargs=gp_kwargs, seed=100
    )
    gp = ComponentGP(**gp_kwargs)
    idx = adaptive_example["selected_indices"]
    gp.fit(example.time[idx], example.full_observation[idx])
    mean, std = gp.predict(dense_grid)

    plt.figure(figsize=(10, 5.5))
    plt.plot(example.time, example.latent, label="Underlying trajectory")
    plt.scatter(
        example.time[idx], example.full_observation[idx],
        s=18, label="FALCON measurements"
    )
    plt.plot(dense_grid, mean, label="GP posterior mean")
    plt.fill_between(
        dense_grid, mean - 2*std, mean + 2*std,
        alpha=0.18, label="Approx. ±2σ"
    )
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.title(f"FALCON Adaptive Trajectory — {example.component_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "trajectory_comparison.png", dpi=160)
    plt.close()

    names = [r["component"] for r in rows]
    fixed_counts = [r["fixed_measurements"] for r in rows]
    adaptive_counts = [r["falcon_measurements"] for r in rows]
    x = np.arange(len(names))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, fixed_counts, width, label="Fixed-rate")
    plt.bar(x + width/2, adaptive_counts, width, label="FALCON adaptive")
    plt.xticks(x, names)
    plt.ylabel("Number of measurements")
    plt.title("Measurement Count Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "measurement_counts.png", dpi=160)
    plt.close()

    print("\nFALCON SOFTWARE MVP")
    print("===================")
    print(f"Threshold: {threshold:.4f}")
    for r in rows:
        print(
            f"{r['component']}: truth={r['ground_truth']:<9} "
            f"fixed={r['fixed_decision']:<9} "
            f"FALCON={r['falcon_decision']:<9} "
            f"reads={r['falcon_measurements']}/{r['fixed_measurements']}"
        )
    print(f"\nFixed measurements:   {fixed_total}")
    print(f"FALCON measurements:  {falcon_total}")
    print(f"Simulation reduction: {reduction*100:.2f}%")
    print(f"Decision agreement:   {agreement*100:.2f}%")
    print("\nResults saved in results/")


if __name__ == "__main__":
    main()
