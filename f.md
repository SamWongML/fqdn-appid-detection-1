```
"""
Feature Pipeline Module

Unified pipeline for feature processing combining all feature engineering components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import joblib

from src.features.feature_engineer import (
    FeatureEngineer,
    FQDNFeatureExtractor,
    TextVectorizer,
    CategoricalEncoder,
)
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.helpers import timer

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
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        tfidf_max_features: int = 500,
        categorical_strategy: str = "label",
        feature_selection_threshold: Optional[float] = None,
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
            "record_type", "fqdn_source", "fqdn_status",
            "brand", "product", "market", "tech_environment",
            "country_code", "itso_id", "buslevel4", "buslevel5",
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
        self._selected_features_mask: Optional[np.ndarray] = None
        self._selected_feature_names: List[str] = []

        self._fitted = False

    @timer("Fitting feature pipeline")
    def fit(
        self,
        df: pl.DataFrame,
        y: Optional[np.ndarray] = None,
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
        feature_names: List[str],
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
            name for name, selected in zip(feature_names, self._selected_features_mask)
            if selected
        ]

        logger.info(
            f"Selected {sum(self._selected_features_mask)}/{len(feature_names)} features"
        )

    @timer("Transforming features")
    def transform(self, df: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
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
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform DataFrame."""
        self.fit(df, y)
        return self.transform(df)

    @property
    def feature_names(self) -> List[str]:
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
    ) -> Dict[str, float]:
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

    def save(self, path: Union[str, Path]) -> None:
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
    def load(cls, path: Union[str, Path]) -> "FeaturePipeline":
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
    config: Optional[Dict[str, Any]] = None,
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
```

```
"""
Feature Selection Module

Provides feature selection methods including:
- Variance-based filtering
- Correlation-based filtering
- Importance-based selection
- Recursive feature elimination
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    f_classif,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """
    Feature selection pipeline.

    Combines multiple selection strategies:
    1. Variance threshold (remove low-variance features)
    2. Correlation filtering (remove highly correlated features)
    3. Importance-based selection (keep most important)
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        max_features: Optional[int] = None,
        importance_method: Literal["mutual_info", "f_classif", "model"] = "mutual_info",
    ):
        """
        Initialize feature selector.

        Args:
            variance_threshold: Minimum variance to keep feature
            correlation_threshold: Maximum correlation to keep both features
            max_features: Maximum features to keep (None for no limit)
            importance_method: Method for importance-based selection
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.importance_method = importance_method

        # State
        self._variance_selector = None
        self._selected_features: Optional[List[int]] = None
        self._correlation_mask: Optional[np.ndarray] = None
        self._importance_selector = None
        self._feature_importances: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureSelector":
        """
        Fit feature selector.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Optional feature names

        Returns:
            self
        """
        logger.info(f"Fitting feature selector on {X.shape[1]} features...")

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        current_mask = np.ones(X.shape[1], dtype=bool)

        # Step 1: Variance threshold
        if self.variance_threshold > 0:
            self._variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            self._variance_selector.fit(X)
            variance_mask = self._variance_selector.get_support()

            n_removed = (~variance_mask).sum()
            if n_removed > 0:
                logger.info(f"Variance filter removed {n_removed} features")

            current_mask &= variance_mask

        # Step 2: Correlation filtering
        if self.correlation_threshold < 1.0:
            X_filtered = X[:, current_mask]
            corr_mask = self._compute_correlation_mask(X_filtered)

            # Map back to original indices
            filtered_indices = np.where(current_mask)[0]
            removed_indices = filtered_indices[~corr_mask]

            n_removed = len(removed_indices)
            if n_removed > 0:
                logger.info(f"Correlation filter removed {n_removed} features")
                current_mask[removed_indices] = False

            self._correlation_mask = corr_mask

        # Step 3: Importance-based selection
        if self.max_features and self.max_features < current_mask.sum():
            X_filtered = X[:, current_mask]

            if self.importance_method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=self.max_features)
            elif self.importance_method == "f_classif":
                selector = SelectKBest(f_classif, k=self.max_features)
            else:
                selector = SelectKBest(f_classif, k=self.max_features)

            selector.fit(X_filtered, y)
            importance_mask = selector.get_support()

            # Map back to original indices
            filtered_indices = np.where(current_mask)[0]
            removed_indices = filtered_indices[~importance_mask]
            current_mask[removed_indices] = False

            self._importance_selector = selector
            self._feature_importances = selector.scores_

            logger.info(f"Importance filter kept {self.max_features} features")

        self._selected_features = np.where(current_mask)[0].tolist()
        self._fitted = True

        logger.info(
            f"Feature selection complete: {len(self._selected_features)}/{X.shape[1]} features kept"
        )

        return self

    def _compute_correlation_mask(self, X: np.ndarray) -> np.ndarray:
        """Compute mask for correlated features."""
        n_features = X.shape[1]
        mask = np.ones(n_features, dtype=bool)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Handle NaN correlations
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Identify highly correlated pairs
        for i in range(n_features):
            if not mask[i]:
                continue

            for j in range(i + 1, n_features):
                if not mask[j]:
                    continue

                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove the feature with lower variance
                    var_i = X[:, i].var()
                    var_j = X[:, j].var()

                    if var_i >= var_j:
                        mask[j] = False
                    else:
                        mask[i] = False
                        break

        return mask

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature matrix using fitted selector.

        Args:
            X: Feature matrix

        Returns:
            Filtered feature matrix
        """
        if not self._fitted:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        return X[:, self._selected_features]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> List[int]:
        """Get indices of selected features."""
        if not self._fitted:
            raise RuntimeError("Selector not fitted")
        return self._selected_features

    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features."""
        if not self._fitted:
            raise RuntimeError("Selector not fitted")
        return [self._feature_names[i] for i in self._selected_features]

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances if available."""
        if self._feature_importances is None:
            return None

        return {
            name: score
            for name, score in zip(
                [self._feature_names[i] for i in self._selected_features],
                self._feature_importances,
            )
        }


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    max_features: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], FeatureSelector]:
    """
    Convenience function for feature selection.

    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Optional feature names
        variance_threshold: Minimum variance
        correlation_threshold: Maximum correlation
        max_features: Maximum features to keep

    Returns:
        Tuple of (selected features, selected names, fitted selector)
    """
    selector = FeatureSelector(
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        max_features=max_features,
    )

    X_selected = selector.fit_transform(X, y, feature_names)
    selected_names = selector.get_selected_feature_names()

    return X_selected, selected_names, selector

```
