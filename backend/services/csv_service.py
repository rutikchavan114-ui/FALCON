import pandas as pd


REQUIRED_COLUMNS = {
    "component_id",
    "time",
    "value",
}


def validate_csv(file_path: str):
    """
    Validate an uploaded FALCON CSV dataset.

    Required columns:
        component_id
        time
        value
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    if df.empty:
        raise ValueError("CSV file contains no data.")

    if df["component_id"].isnull().any():
        raise ValueError("component_id contains empty values.")

    if df["time"].isnull().any():
        raise ValueError("time contains empty values.")

    if df["value"].isnull().any():
        raise ValueError("value contains empty values.")

    try:
        df["time"] = pd.to_numeric(df["time"])
        df["value"] = pd.to_numeric(df["value"])
    except Exception:
        raise ValueError(
            "time and value columns must contain numeric values."
        )

    if not df["time"].is_monotonic_increasing:
        df = df.sort_values(
            ["component_id", "time"]
        ).reset_index(drop=True)

    return df


def get_dataset_summary(df: pd.DataFrame):
    """
    Generate a summary for the uploaded dataset.
    """

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "components": int(df["component_id"].nunique()),
        "time_min": float(df["time"].min()),
        "time_max": float(df["time"].max()),
        "value_min": float(df["value"].min()),
        "value_max": float(df["value"].max()),
    }