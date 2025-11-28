"""
Base Model Module

Provides abstract base class for all models and model factory.
"""



from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all classification models.

    Defines common interface for:
    - Training
    - Prediction
    - Probability estimation
    - Model persistence
    """

    def __init__(
        self,
        n_classes: int,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize base model.

        Args:
            n_classes: Number of classes
            random_state: Random seed
            **kwargs: Additional model-specific parameters
        """
        self.n_classes = n_classes
        self.random_state = random_state
        self.model = None
        self._fitted = False
        self._feature_names: list[str | None] = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str | None] = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names for importance tracking
            **kwargs: Additional training parameters

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        pass

    def predict_top_k(
        self,
        X: np.ndarray,
        k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict top-k class labels with probabilities.

        Args:
            X: Feature matrix
            k: Number of top predictions

        Returns:
            Tuple of (top-k labels, top-k probabilities)
        """
        probas = self.predict_proba(X)

        # Get top-k indices
        top_k_indices = np.argsort(probas, axis=1)[:, -k:][:, ::-1]

        # Get corresponding probabilities
        top_k_probas = np.take_along_axis(probas, top_k_indices, axis=1)

        return top_k_indices, top_k_probas

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    def save(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        import joblib

        state = {
            "model": self.model,
            "n_classes": self.n_classes,
            "random_state": self.random_state,
            "_fitted": self._fitted,
            "_feature_names": self._feature_names,
        }

        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from file.

        Args:
            path: Path to load model from

        Returns:
            Loaded model instance
        """
        import joblib

        state = joblib.load(path)

        instance = cls.__new__(cls)
        instance.model = state["model"]
        instance.n_classes = state["n_classes"]
        instance.random_state = state["random_state"]
        instance._fitted = state["_fitted"]
        instance._feature_names = state["_feature_names"]

        logger.info(f"Model loaded from {path}")
        return instance

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted

    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )


class ModelFactory:
    """
    Factory for creating model instances.

    Supports registration of custom models and configuration-based instantiation.
    """

    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[BaseModel]) -> None:
        """
        Register a model class.

        Args:
            name: Model name
            model_class: Model class
        """
        cls._registry[name.lower()] = model_class
        logger.debug(f"Registered model: {name}")

    @classmethod
    def create(
        cls,
        name: str,
        n_classes: int,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Create a model instance.

        Args:
            name: Model name
            n_classes: Number of classes
            **kwargs: Model parameters

        Returns:
            Model instance
        """
        name_lower = name.lower()

        if name_lower not in cls._registry:
            raise ValueError(
                f"Unknown model: {name}. " f"Available: {list(cls._registry.keys())}"
            )

        model_class = cls._registry[name_lower]
        return model_class(n_classes=n_classes, **kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        """Get list of available model names."""
        return list(cls._registry.keys())


# Auto-register models on import
def _register_models():
    """Register all built-in models."""
    try:
        from src.models.xgb_model import XGBoostModel

        ModelFactory.register("xgboost", XGBoostModel)
    except ImportError:
        logger.warning("XGBoost not available")

    try:
        from src.models.lgb_model import LightGBMModel

        ModelFactory.register("lightgbm", LightGBMModel)
    except ImportError:
        logger.warning("LightGBM not available")

    try:
        from src.models.catboost_model import CatBoostModel

        ModelFactory.register("catboost", CatBoostModel)
    except ImportError:
        logger.warning("CatBoost not available")

    try:
        from src.models.ensemble import EnsembleModel

        ModelFactory.register("ensemble", EnsembleModel)
    except ImportError:
        logger.warning("Ensemble model not available")


# Register models when module is imported
_register_models()
