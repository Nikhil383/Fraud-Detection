"""Exploratory Data Analysis utilities supporting Polars and Pandas."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from fraud_capstone.config import (
    CLASS_BALANCE_PLOT,
    FRAUD_BY_TYPE_CSV,
    REPORTS_DIR,
    TARGET_COLUMN,
)


def compute_class_imbalance(df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
    """Calculate fraud and legitimate transaction counts and proportions."""
    if isinstance(df, pl.DataFrame):
        total = df.height
        fraud_count = int(df.filter(pl.col(TARGET_COLUMN) == 1).height)
        legit_count = total - fraud_count
        fraud_rate = fraud_count / total if total > 0 else 0.0
        return {
            "total_transactions": total,
            "legit_count": legit_count,
            "fraud_count": fraud_count,
            "fraud_prevalence": fraud_rate,
            "imbalance_ratio": (legit_count / fraud_count) if fraud_count > 0 else float("inf"),
        }

    total = len(df)
    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    legit_count = int((df[TARGET_COLUMN] == 0).sum())
    fraud_rate = fraud_count / total if total > 0 else 0.0

    return {
        "total_transactions": total,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_prevalence": fraud_rate,
        "imbalance_ratio": (legit_count / fraud_count) if fraud_count > 0 else float("inf"),
    }


def compute_fraud_by_transaction_type(df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
    """Analyze fraud counts, total volume, and fraud rates across transaction types."""
    if isinstance(df, pl.DataFrame):
        summary = (
            df.group_by("type")
            .agg(
                pl.len().alias("total_count"),
                pl.col(TARGET_COLUMN).sum().alias("fraud_count"),
                pl.col("amount").sum().alias("total_amount"),
                pl.col("amount").filter(pl.col(TARGET_COLUMN) == 1).sum().alias("fraud_amount"),
                pl.col(TARGET_COLUMN).mean().alias("fraud_rate"),
            )
            .with_columns((pl.col("fraud_rate") * 100.0).alias("fraud_rate_pct"))
            .sort("type")
        )
        return summary

    grouped = (
        df.groupby("type")
        .agg(
            total_count=("step", "count"),
            fraud_count=(TARGET_COLUMN, lambda x: (x == 1).sum()),
            total_amount=("amount", "sum"),
            fraud_amount=("amount", lambda x: x[df.loc[x.index, TARGET_COLUMN] == 1].sum()),
        )
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["total_count"]
    grouped["fraud_rate_pct"] = grouped["fraud_rate"] * 100.0
    return grouped


def plot_class_balance(
    imbalance_stats: Dict[str, Any], output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Plot and save class distribution on log-scale."""
    target_path = Path(output_path) if output_path else CLASS_BALANCE_PLOT
    target_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Legitimate (0)", "Fraud (1)"]
    counts = [imbalance_stats["legit_count"], imbalance_stats["fraud_count"]]
    colors = ["#2b5c8f", "#d9534f"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("PaySim Transaction Class Distribution")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()
    return target_path


def generate_eda_reports(df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
    """Execute complete EDA pipeline and save artifacts."""
    imbalance_stats = compute_class_imbalance(df)
    plot_class_balance(imbalance_stats)

    fraud_by_type = compute_fraud_by_transaction_type(df)
    if isinstance(fraud_by_type, pl.DataFrame):
        fraud_by_type.write_csv(FRAUD_BY_TYPE_CSV)
    else:
        fraud_by_type.to_csv(FRAUD_BY_TYPE_CSV, index=False)

    return {
        "imbalance_stats": imbalance_stats,
        "fraud_by_type": fraud_by_type.to_dicts() if isinstance(fraud_by_type, pl.DataFrame) else fraud_by_type.to_dict(orient="records"),
    }
