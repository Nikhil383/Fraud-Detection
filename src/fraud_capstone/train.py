"""Training pipeline with Polars and Pandas support, multi-model selection (HistGradientBoosting / XGBoost / Random Forest), and validation threshold tuning."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from fraud_capstone.config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    MODEL_FILE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from fraud_capstone.data import (
    load_raw_data_polars,
    save_data_quality_report,
    validate_dataset_polars,
)
from fraud_capstone.eda import generate_eda_reports
from fraud_capstone.evaluate import (
    evaluate_predictions,
    find_optimal_threshold,
    plot_confusion_matrix,
    save_decision_threshold,
    save_metrics_report,
)
from fraud_capstone.features import (
    NOTEBOOK_FEATURES,
    align_features,
    engineer_features,
    get_feature_names,
    save_feature_schema,
)
from fraud_capstone.split import chronological_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(
    data_path: Optional[Path] = None,
    nrows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    model_type: str = "hist_gb",  # "hist_gb", "random_forest", "xgboost", or "logistic"
    filter_transfer_cashout: bool = True,
    tracking: bool = False,
    experiment_name: str = "paysim-fraud-detection",
) -> Dict[str, Any]:
    """Execute the end-to-end training pipeline matching the notebook.

    1. Load data via Polars
    2. Run data quality checks & generate EDA reports
    3. Filter to TRANSFER / CASH_OUT (the only fraudulent transaction types in PaySim)
    4. Chronological split (70% train, 15% val, 15% test)
    5. Feature engineering using Polars expressions
    6. Train balanced classifier (HistGradientBoosting, XGBoost, or RF)
    7. Select optimal threshold on validation set
    8. Evaluate on held-out test set
    9. Save model, threshold, schema, and metrics
    """
    logger.info("Starting PaySim Fraud Detection Pipeline...")
    df = load_raw_data_polars(file_path=data_path, nrows=nrows, sample_frac=sample_frac)
    logger.info(f"Loaded dataset with {df.height:,} rows.")

    # 1. Validation & Data Quality
    quality_summary = validate_dataset_polars(df)
    save_data_quality_report(quality_summary)
    logger.info("Saved data quality report.")

    # 2. EDA reports
    generate_eda_reports(df)
    logger.info("Generated EDA reports and figures.")

    # 3. Filter subset: Only TRANSFER and CASH_OUT can be fraud
    if filter_transfer_cashout:
        df_model = df.filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
        logger.info(f"Filtered modeling subset (TRANSFER & CASH_OUT): {df_model.height:,} rows.")
    else:
        df_model = df

    # 4. Chronological Split
    train_df, val_df, test_df = chronological_split(df_model)
    logger.info(
        f"Data split: Train={train_df.height:,}, Val={val_df.height:,}, Test={test_df.height:,}"
    )

    # 5. Feature Engineering
    train_features_df = engineer_features(train_df)
    val_features_df = engineer_features(val_df)
    test_features_df = engineer_features(test_df)

    # Use notebook features schema
    feature_names = [f for f in NOTEBOOK_FEATURES if f in train_features_df.columns]
    save_feature_schema(feature_names)
    logger.info(f"Engineered and saved feature schema: {feature_names}")

    X_train = align_features(train_features_df, feature_names).to_numpy().astype(np.float64)
    y_train = train_df[TARGET_COLUMN].to_numpy()

    X_val = align_features(val_features_df, feature_names).to_numpy().astype(np.float64)
    y_val = val_df[TARGET_COLUMN].to_numpy()

    X_test = align_features(test_features_df, feature_names).to_numpy().astype(np.float64)
    y_test = test_df[TARGET_COLUMN].to_numpy()

    # 6. Classifier Selection & Training
    logger.info(f"Training classifier: {model_type}")
    if model_type == "hist_gb":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=DEFAULT_RANDOM_STATE,
        )
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed. Please install xgboost or use 'hist_gb'.")
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="aucpr",
            random_state=DEFAULT_RANDOM_STATE,
            tree_method="hist",
        )
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=DEFAULT_RANDOM_STATE)),
        ])

    model.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    # 7. Threshold selection on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh, val_score = find_optimal_threshold(y_val, val_probs, metric="f1")
    logger.info(f"Selected validation threshold: {best_thresh:.4f} (Val F1: {val_score:.4f})")
    save_decision_threshold(best_thresh)

    # 8. Evaluate on held-out test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probs, threshold=best_thresh)
    save_metrics_report(test_metrics)
    plot_confusion_matrix(test_metrics["confusion_matrix"])
    logger.info(
        f"Test Results -> PR-AUC: {test_metrics['pr_auc']:.4f}, "
        f"Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}"
    )

    # 9. Save Model artifact (compatible dictionary packaging)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_names,
            "threshold": best_thresh,
            "model_type": model_type,
        },
        MODEL_FILE,
    )
    logger.info(f"Saved trained model package to {MODEL_FILE}")

    # 10. MLflow Tracking (Optional)
    if tracking:
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{model_type}_pipeline"):
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("threshold", best_thresh)
                mlflow.log_metrics({
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "val_f1": val_score,
                })
                logger.info("Logged run and artifacts to MLflow.")
        except Exception as e:
            logger.warning(f"MLflow tracking failed or omitted: {e}")

    return {
        "model": model,
        "feature_names": feature_names,
        "threshold": best_thresh,
        "metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PaySim Fraud Detection Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Path to PaySim CSV dataset",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows for testing/quick iteration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hist_gb",
        choices=["hist_gb", "random_forest", "xgboost", "logistic"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    args = parser.parse_args()

    train_pipeline(
        data_path=Path(args.data),
        nrows=args.nrows,
        model_type=args.model,
        tracking=args.tracking,
    )


if __name__ == "__main__":
    main()
