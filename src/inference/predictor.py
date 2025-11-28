"""
Inference Module

Provides production-ready inference pipeline:
- Single record prediction
- Batch prediction
- Top-k candidates with confidence
- Open-set recognition
- Prediction explanations
"""



from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config.settings import get_settings
from src.data.preprocessor import DataPreprocessor
from src.features.feature_pipeline import FeaturePipeline
from src.models.base_model import BaseModel
from src.utils.helpers import ClassMapping, timer
from src.utils.logger import get_logger
from src.utils.storage import ModelStorage

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Container for a single prediction result."""

    fqdn: str
    predicted_appid: int
    confidence: float
    top_k_predictions: list[tuple[int, float]] = field(default_factory=list)
    is_uncertain: bool = False
    explanation: dict[str, float | None] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fqdn": self.fqdn,
            "predicted_appid": self.predicted_appid,
            "confidence": self.confidence,
            "top_k_predictions": [
                {"appid": appid, "probability": prob}
                for appid, prob in self.top_k_predictions
            ],
            "is_uncertain": self.is_uncertain,
            "explanation": self.explanation,
        }


@dataclass
class BatchPredictionResult:
    """Container for batch prediction results."""

    predictions: list[PredictionResult]
    total_count: int
    uncertain_count: int
    avg_confidence: float

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        records = []
        for pred in self.predictions:
            record = {
                "fqdn": pred.fqdn,
                "predicted_appid": pred.predicted_appid,
                "confidence": pred.confidence,
                "is_uncertain": pred.is_uncertain,
            }
            # Add top-k predictions
            for i, (appid, prob) in enumerate(pred.top_k_predictions):
                record[f"top_{i+1}_appid"] = appid
                record[f"top_{i+1}_prob"] = prob
            records.append(record)

        return pl.DataFrame(records)

    def save_csv(self, path: str | Path) -> None:
        """Save predictions to CSV."""
        df = self.to_dataframe()
        df.write_csv(path)
        logger.info(f"Predictions saved to {path}")


class Predictor:
    """
    Production inference pipeline.

    Handles:
    - Loading trained model and artifacts
    - Preprocessing input data
    - Feature engineering
    - Making predictions with confidence
    - Open-set recognition (unknown class detection)
    """

    def __init__(
        self,
        model: BaseModel,
        preprocessor: DataPreprocessor,
        feature_pipeline: FeaturePipeline,
        class_mapping: ClassMapping,
        confidence_threshold: float = 0.3,
        top_k: int = 5,
    ):
        """
        Initialize predictor.

        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            feature_pipeline: Fitted feature pipeline
            class_mapping: Class label mapping
            confidence_threshold: Threshold for uncertain predictions
            top_k: Number of top predictions to return
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_pipeline = feature_pipeline
        self.class_mapping = class_mapping
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

    @classmethod
    def from_path(
        cls,
        model_path: str | Path,
        model_name: str = "fqdn_classifier",
        version: str | None = None,
    ) -> "Predictor":
        """
        Load predictor from saved model path.

        Args:
            model_path: Base path for model storage
            model_name: Name of the model
            version: Model version (None for latest)

        Returns:
            Predictor instance
        """
        storage = ModelStorage(model_path)
        loaded = storage.load(model_name, version)

        model = loaded["model"]
        artifacts = loaded["artifacts"]

        return cls(
            model=model,
            preprocessor=artifacts.get("preprocessor"),
            feature_pipeline=artifacts.get("feature_pipeline"),
            class_mapping=artifacts.get("class_mapping"),
        )

    def predict_single(
        self,
        fqdn: str,
        record_type: str | None = None,
        record_data: str | None = None,
        additional_features: dict[str, Any | None] = None,
    ) -> PredictionResult:
        """
        Make prediction for a single FQDN.

        Args:
            fqdn: FQDN string
            record_type: DNS record type
            record_data: DNS record data
            additional_features: Additional feature values

        Returns:
            PredictionResult
        """
        # Create DataFrame for single record
        record = {"fqdn": fqdn}
        if record_type:
            record["record_type"] = record_type
        if record_data:
            record["record_data"] = record_data
        if additional_features:
            record.update(additional_features)

        df = pl.DataFrame([record])

        # Get batch prediction
        results = self.predict_batch(df)

        return results.predictions[0]

    @timer("Batch prediction")
    def predict_batch(
        self,
        df: pl.DataFrame,
        include_explanation: bool = False,
    ) -> BatchPredictionResult:
        """
        Make predictions for a batch of records.

        Args:
            df: DataFrame with FQDN records
            include_explanation: Whether to include feature explanations

        Returns:
            BatchPredictionResult
        """
        logger.info(f"Predicting for {len(df)} records...")

        # Preprocess
        if self.preprocessor:
            df = self.preprocessor.transform(df)

        # Extract features
        X, _ = self.feature_pipeline.transform(df)

        # Get predictions
        proba = self.model.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)

        # Get top-k predictions
        top_k_indices = np.argsort(-proba, axis=1)[:, : self.top_k]

        # Get FQDNs
        fqdns = df["fqdn"].to_list()

        # Build results
        results = []
        uncertain_count = 0

        for i in range(len(df)):
            # Decode predicted class
            pred_idx = predictions[i]
            pred_label = self.class_mapping.idx_to_label(pred_idx)

            # Top-k predictions
            top_k = []
            for j in range(self.top_k):
                if j < len(top_k_indices[i]):
                    idx = top_k_indices[i][j]
                    label = self.class_mapping.idx_to_label(idx)
                    prob = float(proba[i, idx])
                    top_k.append((label, prob))

            # Check uncertainty
            is_uncertain = confidences[i] < self.confidence_threshold
            if is_uncertain:
                uncertain_count += 1

            # Explanation (optional)
            explanation = None
            if include_explanation:
                explanation = self._get_explanation(X[i : i + 1], pred_idx)

            result = PredictionResult(
                fqdn=fqdns[i],
                predicted_appid=pred_label,
                confidence=float(confidences[i]),
                top_k_predictions=top_k,
                is_uncertain=is_uncertain,
                explanation=explanation,
            )
            results.append(result)

        return BatchPredictionResult(
            predictions=results,
            total_count=len(results),
            uncertain_count=uncertain_count,
            avg_confidence=float(confidences.mean()),
        )

    def _get_explanation(
        self,
        X: np.ndarray,
        pred_class: int,
        top_n: int = 10,
    ) -> dict[str, float]:
        """Get feature importance explanation for a prediction."""
        try:
            import shap

            # Use TreeExplainer for tree-based models
            if hasattr(self.model, "model"):
                explainer = shap.TreeExplainer(self.model.model)
                shap_values = explainer.shap_values(X)

                if isinstance(shap_values, list):
                    # Multi-class
                    values = shap_values[pred_class][0]
                else:
                    values = shap_values[0]

                # Get feature names
                feature_names = self.feature_pipeline.feature_names

                # Sort by absolute value
                indices = np.argsort(-np.abs(values))[:top_n]

                return {feature_names[i]: float(values[i]) for i in indices}

        except Exception as e:
            logger.debug(f"Explanation failed: {e}")

        return {}

    def predict_with_fallback(
        self,
        df: pl.DataFrame,
        fallback_threshold: float = 0.2,
    ) -> BatchPredictionResult:
        """
        Predict with fallback for very uncertain predictions.

        For predictions below fallback_threshold, returns -1 (unknown).

        Args:
            df: DataFrame with FQDN records
            fallback_threshold: Threshold for unknown class

        Returns:
            BatchPredictionResult
        """
        results = self.predict_batch(df)

        # Mark very uncertain predictions as unknown
        for pred in results.predictions:
            if pred.confidence < fallback_threshold:
                pred.predicted_appid = -1  # Unknown
                pred.is_uncertain = True

        return results


def create_predictor(
    model_path: str | Path = "models/trained",
    model_name: str = "fqdn_classifier",
    version: str | None = None,
) -> Predictor:
    """
    Factory function to create a predictor.

    Args:
        model_path: Path to model storage
        model_name: Name of the model
        version: Model version

    Returns:
        Predictor instance
    """
    return Predictor.from_path(model_path, model_name, version)
