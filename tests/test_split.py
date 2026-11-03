"""Tests for chronological dataset splitting supporting Pandas and Polars."""

import pandas as pd
import polars as pl
import pytest

from fraud_capstone.split import chronological_split


def test_chronological_split_ordering_and_sizes():
    """Verify that chronological splitting respects time steps and ratio partitions in pandas."""
    data = []
    for step in range(1, 101):
        data.append({"step": step, "amount": float(step * 10), "isFraud": 0})
    df = pd.DataFrame(data)

    train_df, val_df, test_df = chronological_split(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check temporal ordering
    assert train_df["step"].max() <= val_df["step"].min()
    assert val_df["step"].max() <= test_df["step"].min()


def test_chronological_split_polars():
    """Verify chronological splitting with Polars DataFrames."""
    pl_df = pl.DataFrame({
        "step": list(range(1, 101)),
        "amount": [float(i * 10) for i in range(1, 101)],
        "isFraud": [0] * 100,
    })

    train_df, val_df, test_df = chronological_split(
        pl_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    assert train_df.height == 70
    assert val_df.height == 15
    assert test_df.height == 15
    assert train_df["step"].max() <= val_df["step"].min()
    assert val_df["step"].max() <= test_df["step"].min()


def test_chronological_split_invalid_ratios():
    """Verify ValueError is raised when ratios do not sum to 1.0."""
    df = pd.DataFrame({"step": [1, 2, 3]})
    with pytest.raises(ValueError):
        chronological_split(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)
