"""
Feature Selection Module

Provides feature selection methods including:
- Variance-based filtering
- Correlation-based filtering
- Importance-based selection
- Recursive feature elimination
"""



from typing import Any, Literal

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
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
        max_features: int | None = None,
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
        self._selected_features: list[int | None] = None
        self._correlation_mask: np.ndarray | None = None
        self._importance_selector = None
        self._feature_importances: np.ndarray | None = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str | None] = None,
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
            self._variance_selector = VarianceThreshold(
                threshold=self.variance_threshold
            )
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
        feature_names: list[str | None] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> list[int]:
        """Get indices of selected features."""
        if not self._fitted:
            raise RuntimeError("Selector not fitted")
        return self._selected_features

    def get_selected_feature_names(self) -> list[str]:
        """Get names of selected features."""
        if not self._fitted:
            raise RuntimeError("Selector not fitted")
        return [self._feature_names[i] for i in self._selected_features]

    def get_feature_importances(self) -> dict[str, float | None]:
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
    feature_names: list[str | None] = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    max_features: int | None = None,
) -> tuple[np.ndarray, list[str], FeatureSelector]:
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
