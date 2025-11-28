"""
FastAPI Application for FQDN Classification

Provides REST API endpoints for:
- Single FQDN prediction
- Batch prediction
- Model information
- Health checks
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import polars as pl
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predictor import BatchPredictionResult, PredictionResult, Predictor
from src.utils.logger import get_logger, setup_logger
from src.utils.storage import ModelStorage

# Setup logging
setup_logger(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# Global predictor instance
predictor: Predictor | None = None


# ============================================
# Request/Response Models
# ============================================


class SinglePredictionRequest(BaseModel):
    """Request for single FQDN prediction."""

    fqdn: str = Field(..., description="FQDN to classify")
    record_type: str | None = Field(None, description="DNS record type")
    record_data: str | None = Field(None, description="DNS record data")
    include_explanation: bool = Field(False, description="Include feature explanations")


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""

    fqdns: list[dict[str, Any]] = Field(..., description="List of FQDN records")
    include_explanation: bool = Field(False, description="Include feature explanations")


class PredictionResponse(BaseModel):
    """Response for single prediction."""

    fqdn: str
    predicted_appid: int
    confidence: float
    top_k_predictions: list[dict[str, Any]]
    is_uncertain: bool
    explanation: dict[str, float | None] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""

    predictions: list[PredictionResponse]
    total_count: int
    uncertain_count: int
    avg_confidence: float
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    """Response for model information."""

    model_name: str
    model_version: str | None
    n_features: int
    n_classes: int
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str = "1.0.0"


# ============================================
# Application Setup
# ============================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global predictor

    # Startup
    logger.info("Starting FQDN Classification API...")

    model_path = os.getenv("MODEL_PATH", "models/trained")
    model_name = os.getenv("MODEL_NAME", "fqdn_classifier")
    model_version = os.getenv("MODEL_VERSION")

    try:
        predictor = Predictor.from_path(
            model_path=model_path,
            model_name=model_name,
            version=model_version,
        )
        logger.info(f"Model loaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None

    yield

    # Shutdown
    logger.info("Shutting down API...")


app = FastAPI(
    title="FQDN Classification API",
    description="API for classifying FQDNs to application IDs",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Endpoints
# ============================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor else "degraded",
        model_loaded=predictor is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=os.getenv("MODEL_NAME", "fqdn_classifier"),
        model_version=os.getenv("MODEL_VERSION"),
        n_features=predictor.feature_pipeline.n_features,
        n_classes=predictor.model.n_classes,
        status="ready",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: SinglePredictionRequest):
    """
    Predict application ID for a single FQDN.

    Returns the predicted application ID along with confidence score
    and top-k alternative predictions.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
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
                {"appid": appid, "probability": prob}
                for appid, prob in result.top_k_predictions
            ],
            is_uncertain=result.is_uncertain,
            explanation=result.explanation,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict application IDs for a batch of FQDNs.

    Accepts a list of FQDN records and returns predictions for all.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.fqdns) > 10000:
        raise HTTPException(
            status_code=400, detail="Batch size exceeds maximum (10000)"
        )

    try:
        start_time = time.time()

        # Convert to DataFrame
        df = pl.DataFrame(request.fqdns)

        # Predict
        batch_result = predictor.predict_batch(
            df,
            include_explanation=request.include_explanation,
        )

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    fqdn=p.fqdn,
                    predicted_appid=p.predicted_appid,
                    confidence=p.confidence,
                    top_k_predictions=[
                        {"appid": appid, "probability": prob}
                        for appid, prob in p.top_k_predictions
                    ],
                    is_uncertain=p.is_uncertain,
                    explanation=p.explanation,
                )
                for p in batch_result.predictions
            ],
            total_count=batch_result.total_count,
            uncertain_count=batch_result.uncertain_count,
            avg_confidence=batch_result.avg_confidence,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "FQDN Classification API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ============================================
# Prometheus Metrics (optional)
# ============================================

try:
    from fastapi.responses import PlainTextResponse
    from prometheus_client import Counter, Histogram, generate_latest

    # Metrics
    PREDICTION_COUNT = Counter(
        "fqdn_predictions_total", "Total number of predictions", ["endpoint"]
    )
    PREDICTION_LATENCY = Histogram(
        "fqdn_prediction_latency_seconds", "Prediction latency in seconds", ["endpoint"]
    )

    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(generate_latest(), media_type="text/plain")

except ImportError:
    logger.info("Prometheus client not available, metrics disabled")


# ============================================
# Run with uvicorn
# ============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "1")),
        reload=os.getenv("ENVIRONMENT", "production") == "development",
    )
