"""
Base Model Module

Provides unified interface for different ML models:
- XGBoost
- LightGBM
- CatBoost

All models implement the same interface for easy swapping and ensembling.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config.settings import (
    CatBoostSettings,
    LightGBMSettings,
    XGBoostSettings,
    get_settings,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, name: str = "base_model", **kwargs):
        self.name = name
        self.model = None
        self._fitted = False
        self._n_classes: int | None = None
        self._classes: np.ndarray | None = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "BaseModel":
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted

    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        if self._n_classes is None:
            raise RuntimeError("Model not fitted")
        return self._n_classes

    @property
    def classes(self) -> np.ndarray:
        """Get class labels."""
        if self._classes is None:
            raise RuntimeError("Model not fitted")
        return self._classes

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        raise AttributeError("Model doesn't have feature importances")

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "model": self.model,
            "_fitted": self._fitted,
            "_n_classes": self._n_classes,
            "_classes": self._classes,
        }
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load model from disk."""
        state = joblib.load(path)
        instance = cls(name=state["name"])
        instance.model = state["model"]
        instance._fitted = state["_fitted"]
        instance._n_classes = state["_n_classes"]
        instance._classes = state["_classes"]
        logger.info(f"Model loaded from {path}")
        return instance


class XGBoostModel(BaseModel):
    """XGBoost classifier wrapper."""

    def __init__(
        self,
        name: str = "xgboost",
        settings: XGBoostSettings | None = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if settings is None:
            settings = get_settings().model.xgboost

        self.settings = settings
        self._params = self._build_params(**kwargs)

    def _build_params(self, **kwargs) -> dict[str, Any]:
        """Build XGBoost parameters."""
        params = {
            "objective": self.settings.objective,
            "eval_metric": self.settings.eval_metric,
            "tree_method": self.settings.tree_method,
            "device": self.settings.device,
            "n_estimators": self.settings.n_estimators,
            "max_depth": self.settings.max_depth,
            "min_child_weight": self.settings.min_child_weight,
            "subsample": self.settings.subsample,
            "colsample_bytree": self.settings.colsample_bytree,
            "colsample_bylevel": self.settings.colsample_bylevel,
            "learning_rate": self.settings.learning_rate,
            "gamma": self.settings.gamma,
            "reg_alpha": self.settings.reg_alpha,
            "reg_lambda": self.settings.reg_lambda,
            "max_delta_step": self.settings.max_delta_step,
            "scale_pos_weight": self.settings.scale_pos_weight,
            "verbosity": self.settings.verbosity,
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "XGBoostModel":
        """Fit XGBoost model."""
        import xgboost as xgb

        logger.info("Training XGBoost model...")

        # Store class info
        self._classes = np.unique(y_train)
        self._n_classes = len(self._classes)

        # Create model
        self.model = xgb.XGBClassifier(**self._params)

        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Fit with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=self.settings.verbosity > 0,
        )

        self._fitted = True
        best_iter = getattr(self.model, "best_iteration", "N/A")
        logger.info(f"XGBoost training complete. Best iteration: {best_iter}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """LightGBM classifier wrapper."""

    def __init__(
        self,
        name: str = "lightgbm",
        settings: LightGBMSettings | None = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if settings is None:
            settings = get_settings().model.lightgbm

        self.settings = settings
        self._params = self._build_params(**kwargs)

    def _build_params(self, **kwargs) -> dict[str, Any]:
        """Build LightGBM parameters."""
        params = {
            "objective": self.settings.objective,
            "metric": self.settings.metric,
            "boosting_type": self.settings.boosting_type,
            "device": self.settings.device,
            "n_estimators": self.settings.n_estimators,
            "num_leaves": self.settings.num_leaves,
            "max_depth": self.settings.max_depth,
            "min_child_samples": self.settings.min_child_samples,
            "min_child_weight": self.settings.min_child_weight,
            "subsample": self.settings.subsample,
            "subsample_freq": self.settings.subsample_freq,
            "colsample_bytree": self.settings.colsample_bytree,
            "learning_rate": self.settings.learning_rate,
            "reg_alpha": self.settings.reg_alpha,
            "reg_lambda": self.settings.reg_lambda,
            "random_state": 42,
            "n_jobs": self.settings.num_threads,
            "verbose": self.settings.verbose,
            "class_weight": self.settings.class_weight,
            "is_unbalance": self.settings.is_unbalance,
        }
        params.update(kwargs)
        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "LightGBMModel":
        """Fit LightGBM model."""
        import lightgbm as lgb

        logger.info("Training LightGBM model...")

        # Store class info
        self._classes = np.unique(y_train)
        self._n_classes = len(self._classes)

        # Update params for multiclass
        params = self._params.copy()
        params["num_class"] = self._n_classes

        # Create model
        self.model = lgb.LGBMClassifier(**params)

        # Prepare callbacks
        callbacks = []
        if self.settings.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.settings.early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=100))

        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Fit
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self._fitted = True
        best_iter = getattr(self.model, "best_iteration_", "N/A")
        logger.info(f"LightGBM training complete. Best iteration: {best_iter}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel):
    """CatBoost classifier wrapper."""

    def __init__(
        self,
        name: str = "catboost",
        settings: CatBoostSettings | None = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if settings is None:
            settings = get_settings().model.catboost

        self.settings = settings
        self._params = self._build_params(**kwargs)

    def _build_params(self, **kwargs) -> dict[str, Any]:
        """Build CatBoost parameters."""
        params = {
            "loss_function": self.settings.loss_function,
            "eval_metric": self.settings.eval_metric,
            "task_type": self.settings.task_type,
            "iterations": self.settings.iterations,
            "depth": self.settings.depth,
            "min_data_in_leaf": self.settings.min_data_in_leaf,
            "learning_rate": self.settings.learning_rate,
            "l2_leaf_reg": self.settings.l2_leaf_reg,
            "random_strength": self.settings.random_strength,
            "bagging_temperature": self.settings.bagging_temperature,
            "one_hot_max_size": self.settings.one_hot_max_size,
            "random_seed": 42,
            "verbose": self.settings.verbose,
        }
        params.update(kwargs)
        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "CatBoostModel":
        """Fit CatBoost model."""
        from catboost import CatBoostClassifier

        logger.info("Training CatBoost model...")

        # Store class info
        self._classes = np.unique(y_train)
        self._n_classes = len(self._classes)

        # Create model
        self.model = CatBoostClassifier(**self._params)

        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        # Fit
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.settings.early_stopping_rounds,
            verbose=self.settings.verbose,
        )

        self._fitted = True
        best_iter = getattr(self.model, "best_iteration_", "N/A")
        logger.info(f"CatBoost training complete. Best iteration: {best_iter}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)


def create_model(
    model_type: str,
    **kwargs,
) -> BaseModel:
    """
    Factory function to create a model.

    Args:
        model_type: Model type ("xgboost", "lightgbm", "catboost")
        **kwargs: Model parameters

    Returns:
        Model instance
    """
    models = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type](**kwargs)
