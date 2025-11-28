"""
Model Evaluation Module

Provides comprehensive evaluation metrics:
- Classification metrics
- Confusion matrix
- Per-class performance
- Confidence calibration
- Feature importance (SHAP)
"""



from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    top_3_accuracy: float = 0.0
    top_5_accuracy: float = 0.0
    log_loss_value: float | None = None
    confusion_matrix: np.ndarray | None = None
    classification_report_dict: dict[str, Any | None] = None
    per_class_metrics: dict[int, dict[str, float | None]] = None
    feature_importance: dict[str, float | None] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "top_3_accuracy": self.top_3_accuracy,
            "top_5_accuracy": self.top_5_accuracy,
            "log_loss": self.log_loss_value,
        }

    def log_summary(self) -> None:
        """Log evaluation summary."""
        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"Accuracy:       {self.accuracy:.4f}")
        logger.info(f"F1 (macro):     {self.f1_macro:.4f}")
        logger.info(f"F1 (weighted):  {self.f1_weighted:.4f}")
        logger.info(f"Precision:      {self.precision_macro:.4f}")
        logger.info(f"Recall:         {self.recall_macro:.4f}")
        logger.info(f"Top-3 Accuracy: {self.top_3_accuracy:.4f}")
        logger.info(f"Top-5 Accuracy: {self.top_5_accuracy:.4f}")
        if self.log_loss_value:
            logger.info(f"Log Loss:       {self.log_loss_value:.4f}")
        logger.info("=" * 50)


class Evaluator:
    """Model evaluator with comprehensive metrics."""

    def __init__(
        self,
        model: BaseModel,
        feature_names: list[str | None] = None,
        class_names: list[str | None] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            feature_names: Names of features
            class_names: Names of classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compute_shap: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate model on data.

        Args:
            X: Features
            y: True labels
            compute_shap: Whether to compute SHAP values

        Returns:
            EvaluationResult
        """
        logger.info(f"Evaluating model on {len(X)} samples...")

        # Get predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # Basic metrics
        result = EvaluationResult(
            accuracy=accuracy_score(y, y_pred),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            f1_weighted=f1_score(y, y_pred, average="weighted", zero_division=0),
            precision_macro=precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            recall_macro=recall_score(y, y_pred, average="macro", zero_division=0),
        )

        # Top-k accuracy
        n_classes = y_proba.shape[1]
        if n_classes >= 3:
            result.top_3_accuracy = top_k_accuracy_score(y, y_proba, k=3)
        if n_classes >= 5:
            result.top_5_accuracy = top_k_accuracy_score(y, y_proba, k=5)

        # Log loss
        try:
            result.log_loss_value = log_loss(y, y_proba)
        except Exception:
            pass

        # Confusion matrix
        result.confusion_matrix = confusion_matrix(y, y_pred)

        # Classification report
        result.classification_report_dict = classification_report(
            y, y_pred, output_dict=True, zero_division=0
        )

        # Per-class metrics
        result.per_class_metrics = self._compute_per_class_metrics(y, y_pred, y_proba)

        # Feature importance
        try:
            importances = self.model.feature_importances_
            if self.feature_names:
                result.feature_importance = dict(zip(self.feature_names, importances))
            else:
                result.feature_importance = {
                    f"f{i}": v for i, v in enumerate(importances)
                }
        except AttributeError:
            pass

        # SHAP values
        if compute_shap:
            result.feature_importance = self._compute_shap_importance(X, y)

        result.log_summary()

        return result

    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[int, dict[str, float]]:
        """Compute metrics for each class."""
        classes = np.unique(y_true)
        per_class = {}

        for cls in classes:
            mask = y_true == cls
            n_samples = mask.sum()

            if n_samples == 0:
                continue

            cls_pred = y_pred[mask]
            cls_true = y_true[mask]

            # Accuracy for this class
            cls_accuracy = (cls_pred == cls_true).mean()

            # Confidence for correct predictions
            cls_proba = y_proba[mask, cls]
            correct_mask = cls_pred == cls_true

            per_class[int(cls)] = {
                "n_samples": int(n_samples),
                "accuracy": float(cls_accuracy),
                "avg_confidence": float(cls_proba.mean()),
                "correct_confidence": (
                    float(cls_proba[correct_mask].mean()) if correct_mask.any() else 0.0
                ),
            }

        return per_class

    def _compute_shap_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1000,
    ) -> dict[str, float]:
        """Compute SHAP-based feature importance."""
        try:
            import shap

            # Sample data if too large
            if len(X) > n_samples:
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Create explainer based on model type
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "get_booster"
            ):
                # XGBoost
                explainer = shap.TreeExplainer(self.model.model)
            else:
                # Generic
                explainer = shap.Explainer(self.model.predict_proba, X_sample)

            shap_values = explainer(X_sample)

            # Get mean absolute SHAP values
            if isinstance(shap_values.values, list):
                # Multi-class
                importance = np.abs(np.array(shap_values.values)).mean(axis=(0, 2))
            else:
                importance = np.abs(shap_values.values).mean(axis=0)

            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            return {f"f{i}": v for i, v in enumerate(importance)}

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {}

    def get_error_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Analyze prediction errors.

        Args:
            X: Features
            y: True labels
            n_samples: Number of error samples to return

        Returns:
            Error analysis dictionary
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # Find errors
        errors = y_pred != y
        error_indices = np.where(errors)[0]

        # Get confidence for errors
        error_confidences = np.max(y_proba[errors], axis=1)

        # High confidence errors (most interesting)
        high_conf_error_idx = np.argsort(-error_confidences)[:n_samples]

        error_samples = []
        for idx in high_conf_error_idx:
            orig_idx = error_indices[idx]
            error_samples.append(
                {
                    "index": int(orig_idx),
                    "true_label": int(y[orig_idx]),
                    "predicted_label": int(y_pred[orig_idx]),
                    "confidence": float(y_proba[orig_idx].max()),
                    "top_3_predictions": y_proba[orig_idx]
                    .argsort()[-3:][::-1]
                    .tolist(),
                }
            )

        # Error statistics
        return {
            "total_errors": int(errors.sum()),
            "error_rate": float(errors.mean()),
            "avg_error_confidence": float(error_confidences.mean()),
            "high_confidence_errors": error_samples,
            "confusion_pairs": self._get_confusion_pairs(y, y_pred),
        }

    def _get_confusion_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Get most common confusion pairs."""
        errors = y_pred != y_true

        pairs = {}
        for true, pred in zip(y_true[errors], y_pred[errors]):
            key = (int(true), int(pred))
            pairs[key] = pairs.get(key, 0) + 1

        sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])[:top_k]

        return [
            {"true": pair[0], "predicted": pair[1], "count": count}
            for pair, count in sorted_pairs
        ]

    def save_report(self, result: EvaluationResult, path: str | Path) -> None:
        """Save evaluation report to file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metrics": result.to_dict(),
            "classification_report": result.classification_report_dict,
            "per_class_metrics": result.per_class_metrics,
        }

        if result.feature_importance:
            # Sort by importance
            sorted_importance = sorted(
                result.feature_importance.items(), key=lambda x: -x[1]
            )
            report["top_features"] = sorted_importance[:50]

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {path}")


def evaluate_model(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str | None] = None,
) -> EvaluationResult:
    """
    Convenience function to evaluate a model.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        feature_names: Feature names

    Returns:
        EvaluationResult
    """
    evaluator = Evaluator(model, feature_names)
    return evaluator.evaluate(X, y)
