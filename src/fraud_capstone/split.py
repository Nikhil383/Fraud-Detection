"""Chronological data splitting module to prevent temporal leakage, supporting Polars and Pandas."""

from typing import Tuple, Union
import pandas as pd
import polars as pl

from fraud_capstone.config import (
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
)


def chronological_split(
    df: Union[pd.DataFrame, pl.DataFrame],
    train_ratio: float = SPLIT_TRAIN_RATIO,
    val_ratio: float = SPLIT_VAL_RATIO,
    test_ratio: float = SPLIT_TEST_RATIO,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
    """Split dataframe chronologically by step.

    Ties are preserved in their stable order.
    Total ratios must sum to approximately 1.0.

    Returns:
        (train_df, val_df, test_df)
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
        raise ValueError("Ratios for train, val, and test must sum to 1.0.")

    if isinstance(df, pl.DataFrame):
        sorted_df = df.sort("step", maintain_order=True)
        n = sorted_df.height
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return (
            sorted_df.slice(0, train_end),
            sorted_df.slice(train_end, val_end - train_end),
            sorted_df.slice(val_end, n - val_end),
        )

    # Pandas split
    sorted_df = df.sort_values(by="step", kind="mergesort").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    return train_df, val_df, test_df
