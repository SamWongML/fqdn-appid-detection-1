"""
CatBoost Model Module

Wrapper for CatBoost classifier with:
- Native categorical feature handling
- GPU support
- Robust to overfitting
"""



from typing import Any

import numpy as np
from catboost import CatBoostClassifier, Pool

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CatBoostModel(BaseModel):
    """
    CatBoost classifier wrapper.

    Optimized for FQDN classification with:
    - Best-in-class categorical feature handling
    - Ordered boosting for better generalization
    - Built-in regularization
    """

    def __init__(
        self,
        n_classes: int,
        random_state: int = 42,
        iterations: int = 500,
        depth: int = 8,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        min_data_in_leaf: int = 10,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        one_hot_max_size: int = 25,
        early_stopping_rounds: int = 50,
        task_type: str = "CPU",
        verbose: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize CatBoost model.

        Args:
            n_classes: Number of classes
            random_state: Random seed
            iterations: Number of boosting iterations
            depth: Tree depth
            learning_rate: Learning rate
            l2_leaf_reg: L2 regularization
            min_data_in_leaf: Minimum samples in leaf
            random_strength: Random strength for scoring
            bagging_temperature: Bayesian bootstrap temperature
            one_hot_max_size: Max size for one-hot encoding
            early_stopping_rounds: Rounds for early stopping
            task_type: CPU or GPU
            verbose: Verbosity level
            **kwargs: Additional parameters
        """
        super().__init__(n_classes, random_state)

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.one_hot_max_size = one_hot_max_size
        self.early_stopping_rounds = early_stopping_rounds
        self.task_type = task_type
        self.verbose = verbose

        self._best_iteration: int | None = None
        self._cat_features: list[int | None] = None

    def _create_model(self) -> CatBoostClassifier:
        """Create CatBoost model instance."""
        return CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=self.min_data_in_leaf,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            one_hot_max_size=self.one_hot_max_size,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            task_type=self.task_type,
            random_seed=self.random_state,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str | None] = None,
        categorical_features: list[int | None] = None,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "CatBoostModel":
        """
        Train CatBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            categorical_features: Indices of categorical features
            sample_weight: Sample weights
            **kwargs: Additional parameters

        Returns:
            self
        """
        logger.info(
            f"Training CatBoost: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, {self.n_classes} classes"
        )

        self._feature_names = feature_names
        self._cat_features = categorical_features

        # Create Pool
        train_pool = Pool(
            X_train,
            label=y_train,
            weight=sample_weight,
            feature_names=feature_names,
            cat_features=categorical_features,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                X_val,
                label=y_val,
                feature_names=feature_names,
                cat_features=categorical_features,
            )

        # Create and train model
        self.model = self._create_model()

        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
        )

        self._best_iteration = self.model.get_best_iteration()
        self._fitted = True

        logger.info(
            f"CatBoost training complete. Best iteration: {self._best_iteration}"
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

        return self.model.predict(X).flatten().astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix
        """
        self._check_fitted()

        return self.model.predict_proba(X)

    def get_feature_importance(
        self,
        importance_type: str = "FeatureImportance",
    ) -> dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance

        Returns:
            Dictionary of feature importances
        """
        self._check_fitted()

        importance = self.model.get_feature_importance()
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

        # Save CatBoost model separately
        model_path = path.replace(".joblib", "_catboost.cbm")
        self.model.save_model(model_path)

        state = {
            "model_path": model_path,
            "n_classes": self.n_classes,
            "random_state": self.random_state,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "_fitted": self._fitted,
            "_feature_names": self._feature_names,
            "_cat_features": self._cat_features,
            "_best_iteration": self._best_iteration,
        }

        joblib.dump(state, path)
        logger.info(f"CatBoost model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CatBoostModel":
        """Load model."""
        import joblib

        state = joblib.load(path)

        instance = cls(
            n_classes=state["n_classes"],
            random_state=state["random_state"],
        )

        # Load CatBoost model
        instance.model = CatBoostClassifier()
        instance.model.load_model(state["model_path"])

        instance._fitted = state["_fitted"]
        instance._feature_names = state["_feature_names"]
        instance._cat_features = state["_cat_features"]
        instance._best_iteration = state["_best_iteration"]

        logger.info(f"CatBoost model loaded from {path}")
        return instance
