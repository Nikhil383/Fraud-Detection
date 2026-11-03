"""Feature engineering module supporting both Polars and Pandas expressions.

Features derived match the high-performance PaySim pipeline:
- Temporal: step, hour (step % 24), day_of_week ((step // 24) % 7)
- Type indicator: type_TRANSFER (and one-hot indicators for all transaction types)
- Transaction amount: amount, log_amount
- Balances: oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
- Balance discrepancies:
    * errorBalanceOrig (orig_discrepancy): (newbalanceOrig + amount) - oldbalanceOrg
    * errorBalanceDest (dest_discrepancy): (oldbalanceDest + amount) - newbalanceDest
- Account state indicators:
    * origEmptied: oldbalanceOrg > 0 and newbalanceOrig == 0
    * destZeroBefore: oldbalanceDest == 0 and newbalanceDest == 0
    * isMerchantDest: nameDest starts with 'M'
    * amount_to_balance (amount_to_orig_ratio): amount / (oldbalanceOrg + 1.0)
    * is_orig_dest_same: nameOrig == nameDest
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from fraud_capstone.config import (
    FEATURE_SCHEMA_FILE,
    PROHIBITED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRANSACTION_TYPES,
)

# Standard notebook feature set for modeling
NOTEBOOK_FEATURES = [
    "step",
    "hour",
    "type_TRANSFER",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "origEmptied",
    "destZeroBefore",
    "isMerchantDest",
    "amount_to_balance",
]


def add_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Add all engineered columns to a Polars DataFrame using vectorized expressions."""
    has_name_dest = "nameDest" in df.columns
    is_merchant_expr = (
        pl.col("nameDest").str.starts_with("M").cast(pl.Int8).alias("isMerchantDest")
        if has_name_dest
        else pl.lit(0).cast(pl.Int8).alias("isMerchantDest")
    )

    return df.with_columns(
        (pl.col("step") % 24).alias("hour"),
        ((pl.col("step") // 24) % 7).alias("day_of_week"),
        (pl.col("type") == "TRANSFER").cast(pl.Int8).alias("type_TRANSFER"),
        (pl.col("newbalanceOrig") + pl.col("amount") - pl.col("oldbalanceOrg")).alias("errorBalanceOrig"),
        (pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest")).alias("errorBalanceDest"),
        ((pl.col("oldbalanceOrg") > 0) & (pl.col("newbalanceOrig") == 0)).cast(pl.Int8).alias("origEmptied"),
        ((pl.col("oldbalanceDest") == 0) & (pl.col("newbalanceDest") == 0)).cast(pl.Int8).alias("destZeroBefore"),
        is_merchant_expr,
        (pl.col("amount") / (pl.col("oldbalanceOrg") + 1.0)).alias("amount_to_balance"),
    )


def engineer_features(df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
    """Derive features from raw transaction data without target leakage.

    Accepts either pandas DataFrame or polars DataFrame, returns the corresponding type.
    Includes both the notebook's high-signal features and legacy compatibility fields.
    """
    if isinstance(df, pl.DataFrame):
        return add_features_polars(df)

    out = pd.DataFrame(index=df.index)

    # 1. Temporal features
    step = df["step"]
    out["step"] = step
    out["hour"] = step % 24
    out["hour_of_day"] = step % 24
    out["day_of_week"] = (step // 24) % 7

    # 2. Transaction Amount
    out["amount"] = df["amount"].astype(float)
    out["log_amount"] = np.log1p(np.maximum(df["amount"], 0))

    # 3. Account balances
    old_orig = df["oldbalanceOrg"].astype(float)
    new_orig = df["newbalanceOrig"].astype(float)
    old_dest = df["oldbalanceDest"].astype(float)
    new_dest = df["newbalanceDest"].astype(float)

    out["oldbalanceOrg"] = old_orig
    out["newbalanceOrig"] = new_orig
    out["oldbalanceDest"] = old_dest
    out["newbalanceDest"] = new_dest

    # Delta & discrepancies (consistent naming across notebook & capstone)
    out["orig_balance_change"] = new_orig - old_orig
    out["dest_balance_change"] = new_dest - old_dest
    out["errorBalanceOrig"] = (new_orig + out["amount"]) - old_orig
    out["errorBalanceDest"] = (old_dest + out["amount"]) - new_dest
    out["orig_discrepancy"] = out["errorBalanceOrig"]
    out["dest_discrepancy"] = out["errorBalanceDest"]

    # Ratios
    out["amount_to_balance"] = out["amount"] / (old_orig + 1.0)
    out["amount_to_orig_ratio"] = out["amount_to_balance"]
    out["amount_to_dest_ratio"] = out["amount"] / (old_dest + 1.0)

    # Behavioral flags from notebook
    out["origEmptied"] = ((old_orig > 0) & (new_orig == 0)).astype(int)
    out["destZeroBefore"] = ((old_dest == 0) & (new_dest == 0)).astype(int)
    out["is_zero_orig_balance"] = (old_orig == 0).astype(int)
    out["is_zero_dest_balance"] = (old_dest == 0).astype(int)

    if "nameDest" in df.columns:
        out["isMerchantDest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    else:
        out["isMerchantDest"] = 0

    if "nameOrig" in df.columns and "nameDest" in df.columns:
        out["is_orig_dest_same"] = (df["nameOrig"] == df["nameDest"]).astype(int)
    else:
        out["is_orig_dest_same"] = 0

    # Type flags
    out["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)
    for t_type in TRANSACTION_TYPES:
        out[f"type_{t_type}"] = (df["type"] == t_type).astype(int)

    return out


def get_feature_names(engineered_df: Union[pd.DataFrame, pl.DataFrame]) -> List[str]:
    """Extract list of feature column names, verifying no prohibited columns are present."""
    cols = list(engineered_df.columns)
    for prohibited in PROHIBITED_FEATURE_COLUMNS:
        if prohibited in cols:
            raise ValueError(f"Prohibited column '{prohibited}' leaked into model features!")
    return cols


def save_feature_schema(
    feature_names: List[str], schema_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save the ordered feature schema as JSON."""
    target_path = Path(schema_path) if schema_path else FEATURE_SCHEMA_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_names, "count": len(feature_names)}, f, indent=2)
    return target_path


def load_feature_schema(schema_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Load the ordered list of feature column names."""
    target_path = Path(schema_path) if schema_path else FEATURE_SCHEMA_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Feature schema not found at '{target_path}'.")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["features"]


def align_features(
    df_features: Union[pd.DataFrame, pl.DataFrame], expected_schema: List[str]
) -> Union[pd.DataFrame, np.ndarray]:
    """Ensure dataframe matches the expected feature schema in exact column order."""
    if isinstance(df_features, pl.DataFrame):
        missing = [c for c in expected_schema if c not in df_features.columns]
        if missing:
            exprs = [pl.lit(0).alias(c) for c in missing]
            df_features = df_features.with_columns(exprs)
        return df_features.select(expected_schema)

    df_copy = df_features.copy()
    for col in expected_schema:
        if col not in df_copy.columns:
            df_copy[col] = 0
    return df_copy[expected_schema].copy()
