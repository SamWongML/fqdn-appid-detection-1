"""
Data Preprocessing Module

Provides data preprocessing pipeline including:
- Missing value imputation
- Text normalization
- Feature filtering
- Data type conversion
- Class handling for imbalanced data
"""



import re
from typing import Any, Literal

import numpy as np
import polars as pl

from src.config.settings import get_settings
from src.utils.helpers import ClassMapping, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline for FQDN data.

    Handles:
    - Missing value imputation
    - Text normalization
    - Class filtering and grouping
    - Data type conversion
    - Feature selection
    """

    def __init__(
        self,
        target_column: str = "appid",
        orphan_value: int = 0,
        min_samples_per_class: int = 3,
        handle_rare_classes: Literal["drop", "group", "keep"] = "group",
        rare_class_threshold: int = 5,
        exclude_patterns: list[str | None] = None,
        include_status: list[str | None] = None,
    ):
        """
        Initialize preprocessor.

        Args:
            target_column: Name of target column
            orphan_value: Value indicating orphan records
            min_samples_per_class: Minimum samples to keep a class
            handle_rare_classes: How to handle rare classes
            rare_class_threshold: Threshold for rare class grouping
            exclude_patterns: FQDN patterns to exclude
            include_status: Status values to include
        """
        self.target_column = target_column
        self.orphan_value = orphan_value
        self.min_samples_per_class = min_samples_per_class
        self.handle_rare_classes = handle_rare_classes
        self.rare_class_threshold = rare_class_threshold
        self.exclude_patterns = exclude_patterns or []
        self.include_status = include_status

        # State variables (set during fit)
        self.class_mapping: ClassMapping | None = None
        self.valid_classes: set[int | None] = None
        self.rare_class_label: int = -999  # Label for grouped rare classes
        self.feature_columns: list[str] = []
        self._fitted = False

    def fit(self, df: pl.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor on training data.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        logger.info("Fitting preprocessor...")

        # Determine valid classes
        self._fit_class_handling(df)

        # Determine feature columns
        self._fit_feature_columns(df)

        self._fitted = True
        logger.info(
            f"Preprocessor fitted: {len(self.valid_classes)} valid classes, "
            f"{len(self.feature_columns)} feature columns"
        )

        return self

    def _fit_class_handling(self, df: pl.DataFrame) -> None:
        """Fit class handling logic."""
        # Get class counts (excluding orphans)
        labeled_df = df.filter(pl.col(self.target_column) != self.orphan_value)

        class_counts = labeled_df.group_by(self.target_column).count()

        # Filter classes with minimum samples
        valid_class_df = class_counts.filter(
            pl.col("count") >= self.min_samples_per_class
        )
        self.valid_classes = set(valid_class_df[self.target_column].to_list())

        # Handle rare classes
        if self.handle_rare_classes == "group":
            rare_classes = class_counts.filter(
                (pl.col("count") < self.rare_class_threshold)
                & (pl.col("count") >= self.min_samples_per_class)
            )
            self.rare_classes = set(rare_classes[self.target_column].to_list())
            logger.info(f"Grouped {len(self.rare_classes)} rare classes")
        else:
            self.rare_classes = set()

        # Create class mapping (excluding orphans)
        all_classes = sorted(self.valid_classes - self.rare_classes)
        if self.handle_rare_classes == "group" and self.rare_classes:
            all_classes.append(self.rare_class_label)

        self.class_mapping = ClassMapping(all_classes)

        logger.info(
            f"Class handling: {len(self.valid_classes)} valid classes, "
            f"{self.class_mapping.n_classes} final classes"
        )

    def _fit_feature_columns(self, df: pl.DataFrame) -> None:
        """Determine feature columns to use."""
        # Exclude target and ID columns
        exclude_cols = {
            self.target_column,
            "id",
            "uuid",
            "dedup_hash",
            "created_at",
            "updated_at",
            "ods_updated_on",
            "ods_load_date",
        }

        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform DataFrame using fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        logger.info(f"Transforming {len(df)} records...")

        # Apply transformations
        df = self._filter_records(df)
        df = self._normalize_text(df)
        df = self._handle_missing_values(df)
        df = self._transform_classes(df)

        logger.info(f"Transformation complete: {len(df)} records")

        return df

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fit and transform DataFrame.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)

    def _filter_records(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter records based on patterns and status."""
        initial_count = len(df)

        # Filter by status
        if self.include_status and "fqdn_status" in df.columns:
            df = df.filter(pl.col("fqdn_status").is_in(self.include_status))

        # Exclude patterns
        if self.exclude_patterns and "fqdn" in df.columns:
            for pattern in self.exclude_patterns:
                df = df.filter(~pl.col("fqdn").str.contains(pattern))

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} records by patterns/status")

        return df

    def _normalize_text(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize text columns."""
        text_columns = ["fqdn", "record_data", "bus_domain", "bus_sub_domain"]

        for col in text_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).str.to_lowercase().str.strip_chars().alias(col)
                )

        # Clean FQDN specifically
        if "fqdn" in df.columns:
            df = df.with_columns(
                pl.col("fqdn").str.strip_chars_end(".").alias("fqdn")
            )  # Remove trailing dot

        return df

    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values."""
        # Strategy: Fill missing with "unknown" for categoricals, -1 for numerics

        for col in df.columns:
            dtype = df[col].dtype

            if dtype in [pl.Utf8, pl.String]:
                # Fill with "unknown"
                df = df.with_columns(pl.col(col).fill_null("unknown").alias(col))
            elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                # Fill with -1 (sentinel value)
                df = df.with_columns(pl.col(col).fill_null(-1).alias(col))
            elif dtype in [pl.Float32, pl.Float64]:
                # Fill with median (or 0 if all null)
                median = df[col].median()
                fill_value = median if median is not None else 0.0
                df = df.with_columns(pl.col(col).fill_null(fill_value).alias(col))

        return df

    def _transform_classes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform target classes."""
        if self.target_column not in df.columns:
            return df

        # Handle rare classes by grouping
        if self.handle_rare_classes == "group" and self.rare_classes:
            df = df.with_columns(
                pl.when(pl.col(self.target_column).is_in(list(self.rare_classes)))
                .then(self.rare_class_label)
                .otherwise(pl.col(self.target_column))
                .alias(self.target_column)
            )

        # Handle dropped classes
        if self.handle_rare_classes == "drop":
            # Filter to only valid classes or orphans
            df = df.filter(
                pl.col(self.target_column).is_in(list(self.valid_classes))
                | (pl.col(self.target_column) == self.orphan_value)
            )

        return df

    def encode_labels(self, labels: pl.Series) -> np.ndarray:
        """
        Encode labels to indices using fitted class mapping.

        Args:
            labels: Series of labels

        Returns:
            Numpy array of encoded indices
        """
        if self.class_mapping is None:
            raise RuntimeError("Class mapping not fitted")

        return np.array(self.class_mapping.transform(labels.to_list()))

    def decode_labels(self, indices: np.ndarray) -> list[int]:
        """
        Decode indices back to original labels.

        Args:
            indices: Array of indices

        Returns:
            List of original labels
        """
        if self.class_mapping is None:
            raise RuntimeError("Class mapping not fitted")

        return self.class_mapping.inverse_transform(indices.tolist())

    def get_feature_columns(
        self,
        include_text: bool = True,
        include_categorical: bool = True,
        include_numerical: bool = True,
    ) -> list[str]:
        """
        Get feature columns by type.

        Args:
            include_text: Include text columns
            include_categorical: Include categorical columns
            include_numerical: Include numerical columns

        Returns:
            List of column names
        """
        text_columns = {"fqdn", "record_data", "raw_fqdn"}
        categorical_columns = {
            "record_type",
            "fqdn_source",
            "fqdn_status",
            "bus_domain",
            "bus_sub_domain",
            "brand",
            "product",
            "market",
            "tech_environment",
            "country_code",
            "category",
            "itso_id",
            "itso_name",
            "buslevel4",
            "buslevel5",
            "buslevel6",
        }
        numerical_columns = {"num_dots", "appserviceid"}

        result = []

        for col in self.feature_columns:
            if col in text_columns and include_text:
                result.append(col)
            elif col in categorical_columns and include_categorical:
                result.append(col)
            elif col in numerical_columns and include_numerical:
                result.append(col)
            elif include_categorical:  # Default unknown columns to categorical
                result.append(col)

        return result

    def save(self, path: str) -> None:
        """Save preprocessor state."""
        import joblib

        state = {
            "target_column": self.target_column,
            "orphan_value": self.orphan_value,
            "min_samples_per_class": self.min_samples_per_class,
            "handle_rare_classes": self.handle_rare_classes,
            "rare_class_threshold": self.rare_class_threshold,
            "exclude_patterns": self.exclude_patterns,
            "include_status": self.include_status,
            "valid_classes": self.valid_classes,
            "rare_classes": self.rare_classes,
            "rare_class_label": self.rare_class_label,
            "feature_columns": self.feature_columns,
            "class_mapping": self.class_mapping,
            "_fitted": self._fitted,
        }

        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from file."""
        import joblib

        state = joblib.load(path)

        preprocessor = cls(
            target_column=state["target_column"],
            orphan_value=state["orphan_value"],
            min_samples_per_class=state["min_samples_per_class"],
            handle_rare_classes=state["handle_rare_classes"],
            rare_class_threshold=state["rare_class_threshold"],
            exclude_patterns=state["exclude_patterns"],
            include_status=state["include_status"],
        )

        preprocessor.valid_classes = state["valid_classes"]
        preprocessor.rare_classes = state["rare_classes"]
        preprocessor.rare_class_label = state["rare_class_label"]
        preprocessor.feature_columns = state["feature_columns"]
        preprocessor.class_mapping = state["class_mapping"]
        preprocessor._fitted = state["_fitted"]

        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


def preprocess_pipeline(
    df: pl.DataFrame,
    preprocessor: DataPreprocessor | None = None,
    fit: bool = True,
) -> tuple[pl.DataFrame, DataPreprocessor]:
    """
    Convenience function to run preprocessing pipeline.

    Args:
        df: DataFrame to preprocess
        preprocessor: Existing preprocessor (creates new if None)
        fit: Whether to fit the preprocessor

    Returns:
        Tuple of (processed DataFrame, preprocessor)
    """
    if preprocessor is None:
        settings = get_settings()
        preprocessor = DataPreprocessor(
            target_column=settings.target.column,
            orphan_value=settings.target.orphan_value,
            min_samples_per_class=settings.data.class_config.min_samples_per_class,
            handle_rare_classes=settings.data.class_config.handle_rare_classes,
            rare_class_threshold=settings.data.class_config.rare_class_threshold,
            exclude_patterns=settings.data.exclude_patterns,
            include_status=settings.data.include_status,
        )

    if fit:
        df = preprocessor.fit_transform(df)
    else:
        df = preprocessor.transform(df)

    return df, preprocessor
