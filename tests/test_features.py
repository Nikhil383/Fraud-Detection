"""Tests for feature engineering and leakage prevention with both Pandas and Polars."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from fraud_capstone.config import PROHIBITED_FEATURE_COLUMNS, TARGET_COLUMN
from fraud_capstone.features import (
    align_features,
    engineer_features,
    get_feature_names,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame([
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C12345",
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 400.0,
            "nameDest": "M54321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
        {
            "step": 2,
            "type": "TRANSFER",
            "amount": 200.0,
            "nameOrig": "C11111",
            "oldbalanceOrg": 200.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C22222",
            "oldbalanceDest": 10.0,
            "newbalanceDest": 210.0,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        },
    ])


def test_no_prohibited_columns_in_features(sample_raw_df):
    """Verify that isFraud, isFlaggedFraud, nameOrig, and nameDest are never in engineered features."""
    features_df = engineer_features(sample_raw_df)
    cols = get_feature_names(features_df)

    for prohibited in PROHIBITED_FEATURE_COLUMNS:
        assert prohibited not in cols, f"Prohibited column {prohibited} was found in features!"
        assert prohibited not in features_df.columns


def test_engineered_features_derived_properly(sample_raw_df):
    """Verify calculation of temporal, delta, and ratio features."""
    features_df = engineer_features(sample_raw_df)

    assert "hour_of_day" in features_df.columns
    assert "day_of_week" in features_df.columns
    assert "errorBalanceOrig" in features_df.columns
    assert "errorBalanceDest" in features_df.columns
    assert "type_TRANSFER" in features_df.columns
    assert "type_PAYMENT" in features_df.columns
    assert "origEmptied" in features_df.columns
    assert "destZeroBefore" in features_df.columns

    # Check payment 1 values
    row0 = features_df.iloc[0]
    assert row0["hour_of_day"] == 1
    assert row0["type_PAYMENT"] == 1
    assert row0["type_TRANSFER"] == 0
    # errorBalanceOrig: (400 + 100) - 500 == 0
    assert row0["errorBalanceOrig"] == 0.0


def test_polars_feature_engineering():
    """Verify Polars feature engineering produces identical values."""
    pl_df = pl.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 200.0],
        "nameOrig": ["C12345", "C11111"],
        "oldbalanceOrg": [500.0, 200.0],
        "newbalanceOrig": [400.0, 0.0],
        "nameDest": ["M54321", "C22222"],
        "oldbalanceDest": [0.0, 10.0],
        "newbalanceDest": [0.0, 210.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    })
    pl_features = engineer_features(pl_df)
    assert "hour" in pl_features.columns
    assert "type_TRANSFER" in pl_features.columns
    assert "errorBalanceOrig" in pl_features.columns
    assert "origEmptied" in pl_features.columns
    assert pl_features["origEmptied"].to_list() == [0, 1]


def test_feature_alignment():
    """Verify that feature schema alignment properly fills missing columns and maintains order."""
    df_raw = pd.DataFrame({"step": [1], "amount": [50.0]})
    schema = ["amount", "step", "type_CASH_OUT"]

    aligned = align_features(df_raw, schema)
    assert list(aligned.columns) == schema
    assert aligned["type_CASH_OUT"].iloc[0] == 0
