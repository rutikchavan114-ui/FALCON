from __future__ import annotations

import os
import tempfile

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.falcon_csv_service import csv_to_components
from falcon.pipeline import (
    build_reference,
    calibrate_threshold,
    adaptive_classification,
    fixed_rate_classification,
)
from falcon.simulator import make_calibration_set


router = APIRouter(
    prefix="/api/analyze",
    tags=["FALCON Analysis"],
)


@router.post("/csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Run the actual FALCON pipeline on an uploaded CSV.

    Expected CSV columns:
        component_id,time,value
    """

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    temp_path = None

    try:
        # --------------------------------------------------
        # Save uploaded CSV temporarily
        # --------------------------------------------------

        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # --------------------------------------------------
        # Convert CSV → ComponentTrajectory objects
        # --------------------------------------------------

        components = csv_to_components(temp_path)

        if len(components) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "FALCON analysis requires at least "
                    "2 components so a peer/reference "
                    "baseline can be constructed."
                ),
            )

        # --------------------------------------------------
        # Common dense prediction grid
        # --------------------------------------------------

        start_time = max(
            float(component.time[0])
            for component in components
        )

        end_time = min(
            float(component.time[-1])
            for component in components
        )

        if end_time <= start_time:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Components must have overlapping "
                    "time ranges."
                ),
            )

        dense_grid = np.linspace(
            start_time,
            end_time,
            101,
        )

        # --------------------------------------------------
        # Build reference baseline
        #
        # For uploaded data we use the other uploaded
        # components as the peer/reference population.
        #
        # Each component is left out of its own reference
        # later during scoring.
        # --------------------------------------------------

        results = []

        for target_index, target in enumerate(components):

            reference_components = [
                component
                for index, component
                in enumerate(components)
                if index != target_index
            ]

            if len(reference_components) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Each component needs at least "
                        "2 peer components for a stable "
                        "reference baseline."
                    ),
                )

            baseline = build_reference(
                reference_components,
                dense_grid,
            )

            threshold = calibrate_threshold(
                reference_components,
                baseline,
                dense_grid,
                percentile=95.0,
            )

            # --------------------------------------------------
            # FALCON adaptive classification
            # --------------------------------------------------

            falcon_result = adaptive_classification(
                target,
                dense_grid,
                baseline,
                threshold,
            )

            # --------------------------------------------------
            # Fixed-rate reference
            # --------------------------------------------------

            fixed_indices = np.arange(
                len(target.time)
            )

            fixed_result = fixed_rate_classification(
                target,
                fixed_indices,
                dense_grid,
                baseline,
                threshold,
            )

            results.append(
                {
                    "component_id": target.component_id,

                    "ground_truth": "UNKNOWN",
                    "truth": "UNKNOWN",

                    "fixed_decision": fixed_result[
                        "decision"
                    ],

                    "falcon_decision": falcon_result[
                        "decision"
                    ],

                    "decision": falcon_result[
                        "decision"
                    ],

                    "fixed_measurements": fixed_result[
                        "measurement_count"
                    ],

                    "falcon_measurements": falcon_result[
                        "measurement_count"
                    ],

                    "total_points": len(
                        target.time
                    ),

                    "fixed_score": fixed_result[
                        "score"
                    ],

                    "falcon_score": falcon_result[
                        "score"
                    ],

                    "anomaly_score": falcon_result[
                        "score"
                    ],

                    "features": falcon_result[
                        "features"
                    ],

                    "selected_indices": falcon_result[
                        "selected_indices"
                    ],

                    "selected_times": falcon_result[
                        "selected_times"
                    ],

                    "schedule_log": falcon_result[
                        "schedule_log"
                    ],

                    "agreement": (
                        fixed_result["decision"]
                        == falcon_result["decision"]
                    ),
                }
            )

        # --------------------------------------------------
        # Aggregate metrics
        # --------------------------------------------------

        total_fixed = sum(
            item["fixed_measurements"]
            for item in results
        )

        total_falcon = sum(
            item["falcon_measurements"]
            for item in results
        )

        agreement_count = sum(
            item["agreement"]
            for item in results
        )

        measurement_reduction = (
            1.0
            - (
                total_falcon
                / max(total_fixed, 1)
            )
        )

        decision_agreement = (
            agreement_count
            / max(len(results), 1)
        )

        # --------------------------------------------------
        # Return dashboard-compatible response
        # --------------------------------------------------

        return {
            "system": "FALCON",
            "status": "analysis_complete",

            "filename": file.filename,

            "threshold": None,

            "components": results,

            "total_fixed_measurements": total_fixed,

            "total_falcon_measurements": total_falcon,

            "fixed_measurements": total_fixed,

            "falcon_measurements": total_falcon,

            "measurement_reduction": (
                measurement_reduction
            ),

            "decision_agreement": (
                decision_agreement
            ),

            "analysis_mode": "uploaded_csv",

            "ground_truth_available": False,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"FALCON CSV analysis failed: {exc}",
        )

    finally:
        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)
