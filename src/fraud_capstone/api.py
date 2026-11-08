"""FastAPI REST service for PaySim transaction fraud scoring."""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fraud_capstone.config import (
    FEATURE_SCHEMA_FILE,
    MODEL_FILE,
    THRESHOLD_FILE,
)
from fraud_capstone.predict import FraudPredictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaySim Fraud Detection API",
    description="Real-time transaction fraud scoring service using XGBoost.",
    version="0.1.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
_predictor: Optional[FraudPredictor] = None


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, description="1 step is 1 hour of time.")
    type: str = Field(..., description="CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., ge=0, description="Amount of transaction")
    nameOrig: str = Field(..., description="Customer who started the transaction")
    oldbalanceOrg: float = Field(..., ge=0, description="Initial balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="New balance after transaction")
    nameDest: str = Field(..., description="Recipient ID")
    oldbalanceDest: float = Field(..., ge=0, description="Initial balance recipient before transaction")
    newbalanceDest: float = Field(..., ge=0, description="New balance recipient after transaction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }
    }


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud_prediction: int
    decision_threshold: float
    status: str = "success"


def get_predictor() -> FraudPredictor:
    """Retrieve or lazily initialize the predictor."""
    global _predictor
    if _predictor is None:
        if not MODEL_FILE.exists() or not FEATURE_SCHEMA_FILE.exists():
            raise FileNotFoundError("Model artifacts not found.")
        _predictor = FraudPredictor(
            model_path=MODEL_FILE,
            schema_path=FEATURE_SCHEMA_FILE,
            threshold_path=THRESHOLD_FILE,
        )
    return _predictor


@app.get("/")
def root():
    return {
        "message": "PaySim Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Health check endpoint checking service and artifact availability."""
    model_loaded = MODEL_FILE.exists() and FEATURE_SCHEMA_FILE.exists()
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_artifact_present": model_loaded,
    }


@app.post("/score", response_model=ScoreResponse, status_code=status.HTTP_200_OK)
def score_transaction(transaction: TransactionRequest):
    """Score a transaction for fraud risk."""
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error loading model: {str(e)}",
        )

    try:
        result = predictor.predict_single(transaction.model_dump())
        return ScoreResponse(**result)
    except Exception as e:
        logger.exception("Error scoring transaction")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error scoring transaction: {str(e)}",
        )
