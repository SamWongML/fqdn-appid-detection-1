"""
FastAPI Application for FQDN Classification

Provides REST API for:
- Single FQDN prediction
- Batch prediction
- Model health check
- Model metadata
"""



import os
import time
from contextlib import asynccontextmanager
from typing import Any

import polars as pl
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predictor import Predictor, create_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global predictor instance
_predictor: Predictor | None = None


class PredictionRequest(BaseModel):
    """Single prediction request."""

    fqdn: str = Field(..., description="Fully qualified domain name")
    record_type: str | None = Field(None, description="DNS record type")
    record_data: str | None = Field(None, description="DNS record data")

    model_config = {
        "json_schema_extra": {
            "example": {"fqdn": "api.dev.example.com", "record_type": "A"}
        }
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    records: list[PredictionRequest] = Field(
        ..., description="List of FQDNs to classify"
    )
    include_explanation: bool = Field(False, description="Include feature explanations")


class PredictionResponse(BaseModel):
    """Single prediction response."""

    fqdn: str
    predicted_appid: int
    confidence: float
    top_k_predictions: list[dict[str, Any]]
    is_uncertain: bool


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
    total_count: int
    uncertain_count: int
    avg_confidence: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str | None
    n_classes: int | None


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    n_classes: int
    n_features: int
    confidence_threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _predictor

    # Startup
    logger.info("Starting FQDN Classification API...")

    model_path = os.getenv("MODEL_ARTIFACTS_PATH", "models/trained")
    model_name = os.getenv("MODEL_NAME", "fqdn_classifier")

    try:
        _predictor = create_predictor(model_path=model_path, model_name=model_name)
        logger.info(f"Model loaded successfully: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _predictor = None

    yield

    # Shutdown
    logger.info("Shutting down FQDN Classification API...")
    _predictor = None


app = FastAPI(
    title="FQDN Classification API",
    description="API for classifying FQDNs to application IDs",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> Predictor:
    """Get predictor instance."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _predictor else "degraded",
        model_loaded=_predictor is not None,
        model_name=_predictor.model.name if _predictor else None,
        n_classes=(
            _predictor.model.n_classes
            if _predictor and _predictor.model._fitted
            else None
        ),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    predictor = get_predictor()

    return ModelInfoResponse(
        model_name=predictor.model.name,
        n_classes=predictor.model.n_classes,
        n_features=predictor.feature_pipeline.n_features,
        confidence_threshold=predictor.confidence_threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict appId for a single FQDN."""
    predictor = get_predictor()

    result = predictor.predict_single(
        fqdn=request.fqdn,
        record_type=request.record_type,
        record_data=request.record_data,
    )

    return PredictionResponse(
        fqdn=result.fqdn,
        predicted_appid=result.predicted_appid,
        confidence=result.confidence,
        top_k_predictions=[
            {"appid": a, "probability": p} for a, p in result.top_k_predictions
        ],
        is_uncertain=result.is_uncertain,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict appId for multiple FQDNs."""
    predictor = get_predictor()

    start_time = time.time()

    # Convert to DataFrame
    records = [
        {"fqdn": r.fqdn, "record_type": r.record_type, "record_data": r.record_data}
        for r in request.records
    ]
    df = pl.DataFrame(records)

    # Predict
    results = predictor.predict_batch(
        df, include_explanation=request.include_explanation
    )

    processing_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                fqdn=p.fqdn,
                predicted_appid=p.predicted_appid,
                confidence=p.confidence,
                top_k_predictions=[
                    {"appid": a, "probability": prob} for a, prob in p.top_k_predictions
                ],
                is_uncertain=p.is_uncertain,
            )
            for p in results.predictions
        ],
        total_count=results.total_count,
        uncertain_count=results.uncertain_count,
        avg_confidence=results.avg_confidence,
        processing_time_ms=processing_time,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FQDN Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
