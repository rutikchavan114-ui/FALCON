


from fastapi import FastAPI
from pydantic import BaseModel

from backend.routes.upload import router as upload_router
from backend.routes.analyze import router as analyze_router
from falcon.simulator import (
    make_calibration_set,
    make_test_set,
)

from falcon.pipeline import (
    build_reference,
    calibrate_threshold,
    fixed_rate_classification,
    adaptive_classification,
)

import numpy as np


# ============================================================
# FALCON APPLICATION
# ============================================================

app = FastAPI(
    title="FALCON API",
    description="Fault Analysis Logic for Component Optimization & Burn-in",
    version="1.0.0",
)


# ============================================================
# ROUTES
# ============================================================

# CSV upload routes
app.include_router(upload_router)
app.include_router(analyze_router)


# ============================================================
# REQUEST MODELS
# ============================================================

class SimulationRequest(BaseModel):
    normal_components: int = 4
    anomalous_components: int = 1


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "system": "FALCON",
        "status": "online",
        "message": (
            "Fault Analysis Logic for Component "
            "Optimization & Burn-in"
        ),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "FALCON Backend",
    }


# ============================================================
# SIMULATION
# ============================================================

@app.post("/api/simulation/run")
def run_simulation(request: SimulationRequest):

    # --------------------------------------------------------
    # TIME GRIDS
    # --------------------------------------------------------

    physical_grid = np.linspace(
        0.0,
        5.0,
        101,
    )

    dense_grid = np.linspace(
        0.0,
        5.0,
        201,
    )

    # --------------------------------------------------------
    # CALIBRATION DATA
    # --------------------------------------------------------

    calibration = make_calibration_set(
        n_normal=30,
        time=physical_grid,
        rng=np.random.default_rng(123),
    )

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test = make_test_set(
        n_normal=request.normal_components,
        n_anomaly=request.anomalous_components,
        time=physical_grid,
        rng=np.random.default_rng(456),
    )

    # --------------------------------------------------------
    # GAUSSIAN PROCESS CONFIGURATION
    # --------------------------------------------------------

    gp_kwargs = {
        "length_scale": 0.75,
        "signal_variance": 4.0,
        "noise_variance": 0.015**2,
    }

    # --------------------------------------------------------
    # BUILD NORMAL REFERENCE
    # --------------------------------------------------------

    baseline = build_reference(
        calibration,
        dense_grid,
        gp_kwargs,
    )

    # --------------------------------------------------------
    # CALIBRATE ANOMALY THRESHOLD
    # --------------------------------------------------------

    threshold = calibrate_threshold(
        calibration,
        baseline,
        dense_grid,
        gp_kwargs,
        percentile=95.0,
    )

    # --------------------------------------------------------
    # FIXED-RATE SAMPLING
    # --------------------------------------------------------

    fixed_indices = np.arange(
        0,
        len(physical_grid),
        1,
    )

    results = []

    # ========================================================
    # COMPONENT ANALYSIS
    # ========================================================

    for component in test:

        # ----------------------------------------------------
        # FIXED-RATE CLASSIFICATION
        # ----------------------------------------------------

        fixed = fixed_rate_classification(
            component,
            fixed_indices,
            dense_grid,
            baseline,
            threshold,
            gp_kwargs,
        )

        # ----------------------------------------------------
        # FALCON ADAPTIVE CLASSIFICATION
        # ----------------------------------------------------

        adaptive = adaptive_classification(
            component,
            dense_grid,
            baseline,
            threshold,
            base_interval=0.05,
            max_interval=0.50,
            p_threshold=0.05,
            n_samples=50,
            gp_kwargs=gp_kwargs,
            seed=100,
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        results.append(
            {
                # Component identity
                "component_id": component.component_id,

                # Ground truth
                "ground_truth": component.label,
                "truth": component.label,

                # Decisions
                "fixed_decision": fixed["decision"],
                "falcon_decision": adaptive["decision"],
                "decision": adaptive["decision"],

                # Measurement counts
                "fixed_measurements": int(
                    fixed["measurement_count"]
                ),

                "falcon_measurements": int(
                    adaptive["measurement_count"]
                ),

                # Total available measurements
                "total_points": len(physical_grid),

                # Scores
                "fixed_score": float(
                    fixed["score"]
                ),

                "falcon_score": float(
                    adaptive["score"]
                ),

                # Frontend-compatible anomaly score
                "anomaly_score": float(
                    adaptive["score"]
                ),
"selected_indices": adaptive["selected_indices"],
"selected_times": adaptive["selected_times"],
"schedule_log": adaptive["schedule_log"],
"features": adaptive["features"],
                # Decision agreement
                "agreement": (
                    fixed["decision"]
                    == adaptive["decision"]
                ),
            }
        )

    # ========================================================
    # TOTAL MEASUREMENTS
    # ========================================================

    fixed_total = sum(
        r["fixed_measurements"]
        for r in results
    )

    falcon_total = sum(
        r["falcon_measurements"]
        for r in results
    )

    # ========================================================
    # MEASUREMENT REDUCTION
    # ========================================================

    reduction = (
        1.0 - falcon_total / fixed_total
        if fixed_total > 0
        else 0.0
    )

    # ========================================================
    # DECISION AGREEMENT
    # ========================================================

    agreement = (
        np.mean(
            [
                r["agreement"]
                for r in results
            ]
        )
        if results
        else 0.0
    )

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {
        "system": "FALCON",

        "status": "analysis_complete",

        "threshold": float(
            threshold
        ),

        "components": results,

        # ----------------------------------------------------
        # Canonical measurement values
        # ----------------------------------------------------

        "total_fixed_measurements": int(
            fixed_total
        ),

        "total_falcon_measurements": int(
            falcon_total
        ),

        # ----------------------------------------------------
        # Frontend-compatible aliases
        # ----------------------------------------------------

        "fixed_measurements": int(
            fixed_total
        ),

        "falcon_measurements": int(
            falcon_total
        ),

        # ----------------------------------------------------
        # Performance metrics
        # ----------------------------------------------------

        # Example:
        # 0.4317 -> 43.17%
        "measurement_reduction": float(
            reduction
        ),

        # Example:
        # 1.0 -> 100%
        "decision_agreement": float(
            agreement
        ),
    }
