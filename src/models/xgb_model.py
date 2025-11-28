"""
XGBoost Model Module

Wrapper for XGBoost classifier with:
- Multi-class classification support
- Early stopping
- Feature importance
- GPU support
"""



from typing import Any

import numpy as np
import xgboost as xgb

from src.config.settings import get_settings
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier wrapper.

    Optimized for multi-class FQDN classification with:
    - Histogram-based tree method for speed
    - Early stopping with validation set
    - Built-in feature importance
    """

    def __init__(
        self,
        n_classes: int,
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        colsample_bylevel: float = 0.8,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        max_delta_step: int = 1,
        tree_method: str = "hist",
        device: str = "cpu",
        early_stopping_rounds: int = 50,
        verbosity: int = 1,
        **kwargs: Any,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_classes: Number of classes
            random_state: Random seed
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            min_child_weight: Minimum sum of instance weight in child
            subsample: Row subsample ratio
            colsample_bytree: Column subsample ratio per tree
            colsample_bylevel: Column subsample ratio per level
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            max_delta_step: Maximum delta step (helps with imbalanced)
            tree_method: Tree construction algorithm
            device: Device to use (cpu/cuda)
            early_stopping_rounds: Rounds for early stopping
            verbosity: Verbosity level
            **kwargs: Additional parameters
        """
        super().__init__(n_classes, random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_delta_step = max_delta_step
        self.tree_method = tree_method
        self.device = device
        self.early_stopping_rounds = early_stopping_rounds
        self.verbosity = verbosity

        self._best_iteration: int | None = None
        self._evals_result: dict[str, Any] = {}

    def _get_params(self) -> dict[str, Any]:
        """Get XGBoost parameters."""
        params = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "eval_metric": "mlogloss",
            "tree_method": self.tree_method,
            "device": self.device,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "max_delta_step": self.max_delta_step,
            "seed": self.random_state,
            "verbosity": self.verbosity,
        }
        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str | None] = None,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "XGBoostModel":
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            sample_weight: Sample weights
            **kwargs: Additional parameters

        Returns:
            self
        """
        logger.info(
            f"Training XGBoost: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, {self.n_classes} classes"
        )

        self._feature_names = feature_names

        # Create DMatrix
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            weight=sample_weight,
            feature_names=feature_names,
        )

        # Evaluation set
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            evals.append((dval, "val"))

        # Get parameters
        params = self._get_params()

        # Train
        self._evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=(
                self.early_stopping_rounds if X_val is not None else None
            ),
            evals_result=self._evals_result,
            verbose_eval=100 if self.verbosity > 0 else False,
        )

        self._best_iteration = self.model.best_iteration
        self._fitted = True

        logger.info(
            f"XGBoost training complete. Best iteration: {self._best_iteration}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        self._check_fitted()

        dtest = xgb.DMatrix(X, feature_names=self._feature_names)
        probas = self.model.predict(dtest)

        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix
        """
        self._check_fitted()

        dtest = xgb.DMatrix(X, feature_names=self._feature_names)
        return self.model.predict(dtest)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance (gain, weight, cover)

        Returns:
            Dictionary of feature importances
        """
        self._check_fitted()

        importance = self.model.get_score(importance_type=importance_type)

        # Normalize
        total = sum(importance.values()) if importance else 1
        return {k: v / total for k, v in importance.items()}

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history (loss per iteration)."""
        return self._evals_result

    @property
    def best_iteration(self) -> int | None:
        """Get best iteration from early stopping."""
        return self._best_iteration

    def save(self, path: str) -> None:
        """Save model."""
        import joblib

        state = {
            "model": self.model,
            "n_classes": self.n_classes,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "device": self.device,
            "_fitted": self._fitted,
            "_feature_names": self._feature_names,
            "_best_iteration": self._best_iteration,
            "_evals_result": self._evals_result,
        }

        joblib.dump(state, path)
        logger.info(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """Load model."""
        import joblib

        state = joblib.load(path)

        instance = cls(
            n_classes=state["n_classes"],
            random_state=state["random_state"],
            n_estimators=state["n_estimators"],
            max_depth=state["max_depth"],
            learning_rate=state["learning_rate"],
        )

        instance.model = state["model"]
        instance._fitted = state["_fitted"]
        instance._feature_names = state["_feature_names"]
        instance._best_iteration = state["_best_iteration"]
        instance._evals_result = state.get("_evals_result", {})

        logger.info(f"XGBoost model loaded from {path}")
        return instance
