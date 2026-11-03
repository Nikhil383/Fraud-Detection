"""Inference and scoring module for PaySim fraud detection supporting Polars, Pandas, and dicts."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl

from fraud_capstone.config import (
    FEATURE_SCHEMA_FILE,
    MODEL_FILE,
    THRESHOLD_FILE,
)
from fraud_capstone.evaluate import load_decision_threshold
from fraud_capstone.features import align_features, engineer_features, load_feature_schema


class FraudPredictor:
    """Predictor encapsulating trained model, feature schema, and decision threshold."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        threshold_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_FILE
        self.schema_path = Path(schema_path) if schema_path else FEATURE_SCHEMA_FILE
        self.threshold_path = Path(threshold_path) if threshold_path else THRESHOLD_FILE

        self.model = None
        self.feature_schema = []
        self.threshold = 0.5
        self._load_artifacts()

    def _load_artifacts(self):
        """Load artifacts from disk.

        Raises:
            FileNotFoundError: If model or feature schema is missing.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Please train the model first using 'python -m fraud_capstone.train'."
            )
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Feature schema not found at '{self.schema_path}'."
            )

        loaded = joblib.load(self.model_path)
        # Support dict format saved by notebook or standalone estimator
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            if "features" in loaded:
                self.feature_schema = loaded["features"]
            if "threshold" in loaded:
                self.threshold = float(loaded["threshold"])
        else:
            self.model = loaded

        if not self.feature_schema and self.schema_path.exists():
            self.feature_schema = load_feature_schema(self.schema_path)
        if self.threshold_path.exists():
            self.threshold = load_decision_threshold(self.threshold_path)

    def predict_dataframe(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Score a batch dataframe of raw transactions (Polars or Pandas)."""
        is_polars = isinstance(df, pl.DataFrame)
        df_pd = df.to_pandas() if is_polars else df

        # In PaySim, only TRANSFER and CASH_OUT can be fraudulent
        mask = df_pd["type"].isin(["TRANSFER", "CASH_OUT"]).values
        probabilities = np.zeros(len(df_pd), dtype=float)

        if mask.any():
            sub_df = df_pd[mask]
            X_eng = engineer_features(sub_df)
            X_aligned = align_features(X_eng, self.feature_schema)
            if isinstance(X_aligned, pl.DataFrame):
                X_matrix = X_aligned.to_numpy().astype(np.float64)
            else:
                X_matrix = X_aligned.values.astype(np.float64)
            probabilities[mask] = self.model.predict_proba(X_matrix)[:, 1]

        predictions = (probabilities >= self.threshold).astype(int)

        if is_polars:
            return (
                df.with_columns(
                    pl.Series("fraud_probability", probabilities),
                    pl.Series("is_fraud_prediction", predictions),
                    pl.lit(self.threshold).alias("decision_threshold"),
                )
            )

        results = df.copy()
        results["fraud_probability"] = probabilities
        results["is_fraud_prediction"] = predictions
        results["decision_threshold"] = self.threshold
        return results

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single transaction record."""
        df = pd.DataFrame([transaction])
        scored_df = self.predict_dataframe(df)
        row = scored_df.iloc[0]

        return {
            "fraud_probability": float(row["fraud_probability"]),
            "is_fraud_prediction": int(row["is_fraud_prediction"]),
            "decision_threshold": float(row["decision_threshold"]),
        }
