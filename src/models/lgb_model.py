"""
LightGBM Model Module

Wrapper for LightGBM classifier with:
- Multi-class classification support
- Early stopping
- Categorical feature handling
- Feature importance
"""



from typing import Any

import lightgbm as lgb
import numpy as np

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM classifier wrapper.

    Optimized for multi-class FQDN classification with:
    - Leaf-wise tree growth for speed
    - Native categorical feature support
    - Early stopping
    """

    def __init__(
        self,
        n_classes: int,
        random_state: int = 42,
        n_estimators: int = 500,
        num_leaves: int = 128,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        min_child_weight: float = 0.001,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        num_threads: int = -1,
        verbose: int = -1,
        **kwargs: Any,
    ):
        """
        Initialize LightGBM model.

        Args:
            n_classes: Number of classes
            random_state: Random seed
            n_estimators: Number of boosting rounds
            num_leaves: Maximum number of leaves
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            min_child_samples: Minimum samples in leaf
            min_child_weight: Minimum sum of hessian in leaf
            subsample: Row subsample ratio
            subsample_freq: Frequency of subsampling
            colsample_bytree: Column subsample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            early_stopping_rounds: Rounds for early stopping
            num_threads: Number of threads (-1 for auto)
            verbose: Verbosity level
            **kwargs: Additional parameters
        """
        super().__init__(n_classes, random_state)

        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.num_threads = num_threads
        self.verbose = verbose

        self._best_iteration: int | None = None
        self._evals_result: dict[str, Any] = {}

    def _get_params(self) -> dict[str, Any]:
        """Get LightGBM parameters."""
        params = {
            "objective": "multiclass",
            "num_class": self.n_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "seed": self.random_state,
            "num_threads": self.num_threads,
            "verbose": self.verbose,
            "force_row_wise": True,  # Suppress warning
        }
        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str | None] = None,
        categorical_features: list[str | None] = None,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "LightGBMModel":
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            categorical_features: Categorical feature names
            sample_weight: Sample weights
            **kwargs: Additional parameters

        Returns:
            self
        """
        logger.info(
            f"Training LightGBM: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, {self.n_classes} classes"
        )

        self._feature_names = feature_names

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight,
            feature_name=feature_names,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=feature_names,
                categorical_feature=categorical_features,
                reference=train_data,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("val")

        # Get parameters
        params = self._get_params()

        # Callbacks
        callbacks = [
            lgb.log_evaluation(period=100),
        ]

        if X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose > 0,
                )
            )

        # Train
        self._evals_result = {}

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._best_iteration = self.model.best_iteration
        self._fitted = True

        logger.info(
            f"LightGBM training complete. Best iteration: {self._best_iteration}"
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

        probas = self.model.predict(X, num_iteration=self._best_iteration)
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

        return self.model.predict(X, num_iteration=self._best_iteration)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance (gain, split)

        Returns:
            Dictionary of feature importances
        """
        self._check_fitted()

        importance = self.model.feature_importance(importance_type=importance_type)
        feature_names = self._feature_names or [f"f{i}" for i in range(len(importance))]

        # Normalize
        total = importance.sum() if importance.sum() > 0 else 1
        return {name: imp / total for name, imp in zip(feature_names, importance)}

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
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "_fitted": self._fitted,
            "_feature_names": self._feature_names,
            "_best_iteration": self._best_iteration,
        }

        joblib.dump(state, path)
        logger.info(f"LightGBM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        """Load model."""
        import joblib

        state = joblib.load(path)

        instance = cls(
            n_classes=state["n_classes"],
            random_state=state["random_state"],
        )

        instance.model = state["model"]
        instance._fitted = state["_fitted"]
        instance._feature_names = state["_feature_names"]
        instance._best_iteration = state["_best_iteration"]

        logger.info(f"LightGBM model loaded from {path}")
        return instance
