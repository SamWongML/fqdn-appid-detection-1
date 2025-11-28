"""
Ensemble Model Module

Provides ensemble methods combining multiple base models:
- Weighted soft voting
- Hard voting
"""



from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from src.config.settings import EnsembleSettings, get_settings
from src.models.base_model import BaseModel, create_model
from src.utils.helpers import timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""

    def __init__(
        self,
        models: list[tuple[str, BaseModel, float | None]] = None,
        voting: Literal["soft", "hard"] = "soft",
        name: str = "ensemble",
        settings: EnsembleSettings | None = None,
    ):
        super().__init__(name=name)
        self.voting = voting
        self.settings = settings or get_settings().model.ensemble

        if models is None:
            self.models: list[tuple[str, BaseModel, float]] = []
            for model_config in self.settings.models:
                model = create_model(model_config.name)
                self.models.append((model_config.name, model, model_config.weight))
        else:
            self.models = models

        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total_weight = sum(w for _, _, w in self.models)
        if total_weight > 0:
            self.models = [
                (name, model, w / total_weight) for name, model, w in self.models
            ]

    @timer("Training ensemble model")
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "EnsembleModel":
        logger.info(f"Training ensemble with {len(self.models)} models...")
        self._classes = np.unique(y_train)
        self._n_classes = len(self._classes)

        for i, (name, model, weight) in enumerate(self.models):
            logger.info(
                f"Training model {i+1}/{len(self.models)}: {name} (weight={weight:.2f})"
            )
            model.fit(X_train, y_train, X_val, y_val, **kwargs)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self._classes[np.argmax(proba, axis=1)]
        else:
            predictions = np.array([model.predict(X) for _, model, _ in self.models])
            weights = np.array([w for _, _, w in self.models])
            result = np.zeros(X.shape[0], dtype=predictions.dtype)
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                unique_votes, indices = np.unique(votes, return_inverse=True)
                weighted_counts = np.bincount(indices, weights=weights)
                result[i] = unique_votes[np.argmax(weighted_counts)]
            return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        probas = [model.predict_proba(X) * weight for _, model, weight in self.models]
        return np.sum(probas, axis=0)

    def predict_with_confidence(
        self, X: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        proba = self.predict_proba(X)
        predictions = self._classes[np.argmax(proba, axis=1)]
        confidences = np.max(proba, axis=1)
        top_k_indices = np.argsort(-proba, axis=1)[:, :top_k]
        top_k_predictions = self._classes[top_k_indices]
        return predictions, confidences, top_k_predictions

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = []
        for name, model, weight in self.models:
            try:
                importances.append(model.feature_importances_ * weight)
            except AttributeError:
                continue
        if not importances:
            raise AttributeError("No model has feature importances")
        return np.sum(importances, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "name": self.name,
            "voting": self.voting,
            "models": self.models,
            "_fitted": self._fitted,
            "_n_classes": self._n_classes,
            "_classes": self._classes,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleModel":
        state = joblib.load(path)
        instance = cls(
            models=state["models"], voting=state["voting"], name=state["name"]
        )
        instance._fitted = state["_fitted"]
        instance._n_classes = state["_n_classes"]
        instance._classes = state["_classes"]
        return instance


def create_ensemble(
    model_configs: list[dict[str, Any | None]] = None, voting: str = "soft"
) -> EnsembleModel:
    if model_configs is None:
        return EnsembleModel(voting=voting)
    models = []
    for config in model_configs:
        model_type = config.pop("type")
        weight = config.pop("weight", 1.0)
        model = create_model(model_type, **config)
        models.append((model_type, model, weight))
    return EnsembleModel(models=models, voting=voting)
