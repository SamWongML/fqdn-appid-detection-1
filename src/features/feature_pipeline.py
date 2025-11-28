"""
Feature Pipeline Module

Unified pipeline for feature processing combining all feature engineering components.
"""



from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from src.config.settings import get_settings
from src.features.feature_engineer import (
    CategoricalEncoder,
    FeatureEngineer,
    FQDNFeatureExtractor,
    TextVectorizer,
)
from src.utils.helpers import timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Complete feature processing pipeline.

    Orchestrates:
    - FQDN feature extraction
    - Text vectorization
    - Categorical encoding
    - Feature selection
    - Scaling and normalization
    """

    def __init__(
        self,
        fqdn_column: str = "fqdn",
        target_column: str = "appid",
        categorical_columns: list[str | None] = None,
        numerical_columns: list[str | None] = None,
        text_columns: list[str | None] = None,
        tfidf_max_features: int = 500,
        categorical_strategy: str = "label",
        feature_selection_threshold: float | None = None,
    ):
        """
        Initialize feature pipeline.

        Args:
            fqdn_column: Name of FQDN column
            target_column: Name of target column
            categorical_columns: Categorical feature columns
            numerical_columns: Numerical feature columns
            text_columns: Text columns for TF-IDF
            tfidf_max_features: Max TF-IDF features
            categorical_strategy: Encoding strategy
            feature_selection_threshold: Feature importance threshold
        """
        self.fqdn_column = fqdn_column
        self.target_column = target_column
        self.categorical_columns = categorical_columns or [
            "record_type",
            "fqdn_source",
            "fqdn_status",
            "brand",
            "product",
            "market",
            "tech_environment",
            "country_code",
            "itso_id",
            "buslevel4",
            "buslevel5",
        ]
        self.numerical_columns = numerical_columns or []
        self.text_columns = text_columns or [fqdn_column]
        self.tfidf_max_features = tfidf_max_features
        self.categorical_strategy = categorical_strategy
        self.feature_selection_threshold = feature_selection_threshold

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(
            fqdn_column=fqdn_column,
            text_columns=self.text_columns,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            tfidf_max_features=tfidf_max_features,
            categorical_strategy=categorical_strategy,
        )

        # Feature selection mask
        self._selected_features_mask: np.ndarray | None = None
        self._selected_feature_names: list[str] = []

        self._fitted = False

    @timer("Fitting feature pipeline")
    def fit(
        self,
        df: pl.DataFrame,
        y: np.ndarray | None = None,
    ) -> "FeaturePipeline":
        """
        Fit the feature pipeline.

        Args:
            df: Training DataFrame
            y: Target values (for feature selection)

        Returns:
            self
        """
        logger.info(f"Fitting feature pipeline on {len(df)} records...")

        # Get target for target encoding if available
        target = None
        if self.target_column in df.columns:
            target = df[self.target_column]

        # Fit feature engineer
        self.feature_engineer.fit(df, target)

        # Transform to get feature matrix
        X, feature_names = self.feature_engineer.transform(df)

        # Feature selection (optional)
        if self.feature_selection_threshold and y is not None:
            self._fit_feature_selection(X, y, feature_names)
        else:
            self._selected_features_mask = np.ones(len(feature_names), dtype=bool)
            self._selected_feature_names = feature_names

        self._fitted = True
        logger.info(
            f"Feature pipeline fitted: {len(self._selected_feature_names)} features selected"
        )

        return self

    def _fit_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Fit feature selection based on importance."""
        from sklearn.ensemble import RandomForestClassifier

        logger.info("Performing feature selection...")

        # Quick random forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X, y)

        importances = rf.feature_importances_

        # Select features above threshold
        self._selected_features_mask = importances >= self.feature_selection_threshold
        self._selected_feature_names = [
            name
            for name, selected in zip(feature_names, self._selected_features_mask)
            if selected
        ]

        logger.info(
            f"Selected {sum(self._selected_features_mask)}/{len(feature_names)} features"
        )

    @timer("Transforming features")
    def transform(self, df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Transform DataFrame to feature matrix.

        Args:
            df: DataFrame to transform

        Returns:
            Tuple of (feature matrix, feature names)
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # Transform with feature engineer
        X, _ = self.feature_engineer.transform(df)

        # Apply feature selection
        if self._selected_features_mask is not None:
            X = X[:, self._selected_features_mask]

        return X, self._selected_feature_names

    def fit_transform(
        self,
        df: pl.DataFrame,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Fit and transform DataFrame."""
        self.fit(df, y)
        return self.transform(df)

    @property
    def feature_names(self) -> list[str]:
        """Get selected feature names."""
        return self._selected_feature_names

    @property
    def n_features(self) -> int:
        """Get number of selected features."""
        return len(self._selected_feature_names)

    def get_feature_importance(
        self,
        model: Any,
        method: str = "native",
    ) -> dict[str, float]:
        """
        Get feature importance from a trained model.

        Args:
            model: Trained model
            method: Importance method ("native", "permutation")

        Returns:
            Dictionary of feature importances
        """
        if method == "native":
            # Try to get native feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                logger.warning("Model doesn't have native feature importance")
                return {}

            return dict(zip(self._selected_feature_names, importances))

        else:
            raise ValueError(f"Unknown method: {method}")

    def save(self, path: str | Path) -> None:
        """Save pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "fqdn_column": self.fqdn_column,
            "target_column": self.target_column,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "text_columns": self.text_columns,
            "tfidf_max_features": self.tfidf_max_features,
            "categorical_strategy": self.categorical_strategy,
            "feature_selection_threshold": self.feature_selection_threshold,
            "feature_engineer": self.feature_engineer,
            "selected_features_mask": self._selected_features_mask,
            "selected_feature_names": self._selected_feature_names,
            "_fitted": self._fitted,
        }

        joblib.dump(state, path)
        logger.info(f"Feature pipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load pipeline from disk."""
        state = joblib.load(path)

        pipeline = cls(
            fqdn_column=state["fqdn_column"],
            target_column=state["target_column"],
            categorical_columns=state["categorical_columns"],
            numerical_columns=state["numerical_columns"],
            text_columns=state["text_columns"],
            tfidf_max_features=state["tfidf_max_features"],
            categorical_strategy=state["categorical_strategy"],
            feature_selection_threshold=state["feature_selection_threshold"],
        )

        pipeline.feature_engineer = state["feature_engineer"]
        pipeline._selected_features_mask = state["selected_features_mask"]
        pipeline._selected_feature_names = state["selected_feature_names"]
        pipeline._fitted = state["_fitted"]

        logger.info(f"Feature pipeline loaded from {path}")
        return pipeline


def create_feature_pipeline(
    config: dict[str, Any | None] = None,
) -> FeaturePipeline:
    """
    Factory function to create feature pipeline from config.

    Args:
        config: Configuration dictionary (uses settings if None)

    Returns:
        FeaturePipeline instance
    """
    if config is None:
        settings = get_settings()
        config = {
            "fqdn_column": "fqdn",
            "target_column": settings.target.column,
            "tfidf_max_features": settings.features.tfidf_max_features,
        }

    return FeaturePipeline(**config)
