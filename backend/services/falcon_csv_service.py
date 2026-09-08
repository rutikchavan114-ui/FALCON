from __future__ import annotations

import numpy as np
import pandas as pd

from falcon.simulator import ComponentTrajectory


REQUIRED_COLUMNS = {
    "component_id",
    "time",
    "value",
}


def csv_to_components(csv_path: str) -> list[ComponentTrajectory]:
    """
    Convert a validated FALCON CSV dataset into
    ComponentTrajectory objects expected by the
    existing FALCON pipeline.
    """

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["component_id"] = df["component_id"].astype(str)
    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce",
    )
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    if df["time"].isna().any():
        raise ValueError(
            "Column 'time' contains invalid numeric values."
        )

    if df["value"].isna().any():
        raise ValueError(
            "Column 'value' contains invalid numeric values."
        )

    components = []

    for component_id, group in df.groupby(
        "component_id",
        sort=False,
    ):
        group = group.sort_values("time")

        time = group["time"].to_numpy(
            dtype=float
        )

        observed = group["value"].to_numpy(
            dtype=float
        )

        if len(time) < 3:
            raise ValueError(
                f"Component {component_id} "
                f"needs at least 3 measurements."
            )

        if not np.all(
            np.diff(time) > 0
        ):
            raise ValueError(
                f"Component {component_id} "
                "must have strictly increasing time values."
            )

        # For real uploaded data we do not know the
        # latent trajectory or true tau.
        #
        # These fields are kept only because the
        # existing ComponentTrajectory structure
        # expects them.
        latent = observed.copy()

        tau = float("nan")

        components.append(
            ComponentTrajectory(
                component_id=component_id,
                time=time,
                latent=latent,
                full_observation=observed,
                tau=tau,
                label="UNKNOWN",
            )
        )

    if not components:
        raise ValueError(
            "CSV contains no component data."
        )

    return components
