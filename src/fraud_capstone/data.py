"""Data loading and validation module supporting both Polars and Pandas."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from fraud_capstone.config import (
    DATA_QUALITY_REPORT,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def resolve_data_path(file_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve data path checking RAW_DATA_PATH or alternate locations."""
    if file_path:
        path = Path(file_path)
    else:
        alt_dataset_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "paysim.csv"
        if RAW_DATA_PATH.exists():
            path = RAW_DATA_PATH
        elif alt_dataset_path.exists():
            path = alt_dataset_path
        else:
            path = RAW_DATA_PATH
    return path


def load_raw_data_polars(
    file_path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pl.DataFrame:
    """Load PaySim CSV data directly as a Polars DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns or targets are invalid.
    """
    path = resolve_data_path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found at '{path}'. "
            "Please place paysim.csv in 'dataset/' or 'data/raw/', or specify via --data."
        )

    df = pl.read_csv(path, n_rows=nrows)
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=random_state)

    validate_dataset_polars(df)
    return df


def validate_dataset_polars(df: pl.DataFrame) -> Dict[str, Any]:
    """Validate required columns, types, and labels for a Polars DataFrame."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required column(s): {missing_cols}")

    unique_labels = set(df[TARGET_COLUMN].drop_nulls().unique().to_list())
    valid_labels = {0, 1}
    if not unique_labels.issubset(valid_labels):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains invalid labels: {unique_labels}. "
            "Expected binary values {0, 1}."
        )

    null_counts = {col: int(df[col].null_count()) for col in df.columns}
    dups = int(df.is_duplicated().sum()) if "row_id" not in df.columns else int(df.drop("row_id").is_duplicated().sum())
    neg_amounts = int((df["amount"] < 0).sum()) if "amount" in df.columns else 0

    class_counts = df[TARGET_COLUMN].value_counts().to_dicts()
    fraud_dist = {int(row[TARGET_COLUMN]): int(row["count"]) for row in class_counts}

    return {
        "num_rows": int(df.height),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values": null_counts,
        "duplicate_rows": dups,
        "negative_amounts": neg_amounts,
        "is_fraud_distribution": fraud_dist,
    }


def load_raw_data(
    file_path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the raw PaySim CSV data as a Pandas DataFrame (for backwards compatibility).

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing or the target is invalid.
    """
    path = resolve_data_path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found at '{path}'. "
            "Please place paysim.csv in 'dataset/' or 'data/raw/', or specify via --data."
        )

    df = pd.read_csv(path, nrows=nrows)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate required columns, types, and labels.

    Raises:
        ValueError: When schema or label constraints are violated.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required column(s): {missing_cols}")

    # Check target values
    unique_labels = set(df[TARGET_COLUMN].dropna().unique())
    valid_labels = {0, 1}
    if not unique_labels.issubset(valid_labels):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains invalid labels: {unique_labels}. "
            "Expected binary values {0, 1}."
        )

    # Compile quality summary
    quality_summary = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_amounts": int((df["amount"] < 0).sum()) if "amount" in df.columns else 0,
        "is_fraud_distribution": {
            int(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()
        },
    }
    return quality_summary


def save_data_quality_report(
    quality_summary: Dict[str, Any], output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save the data quality summary as a JSON report."""
    target_path = Path(output_path) if output_path else DATA_QUALITY_REPORT
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2)
    return target_path
