"""
Data Validation Module

Provides comprehensive data quality validation including:
- Schema validation
- Missing value checks
- Data type verification
- Business rule validation
- Statistical anomaly detection
"""



import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    column: str | None
    issue_type: str
    message: str
    severity: ValidationSeverity
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        """Get info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def add_issue(
        self,
        column: str | None,
        issue_type: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
        details: dict[str, Any | None] = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(
            ValidationIssue(
                column=column,
                issue_type=issue_type,
                message=message,
                severity=severity,
                details=details or {},
            )
        )

        if severity == ValidationSeverity.ERROR:
            self.is_valid = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "issues": [i.to_dict() for i in self.issues],
            "statistics": self.statistics,
        }

    def log_summary(self) -> None:
        """Log validation summary."""
        if self.is_valid:
            logger.info(f"Data validation passed with {len(self.warnings)} warnings")
        else:
            logger.error(f"Data validation failed with {len(self.errors)} errors")

        for issue in self.errors:
            logger.error(f"  [{issue.column}] {issue.message}")

        for issue in self.warnings[:5]:  # Limit warnings shown
            logger.warning(f"  [{issue.column}] {issue.message}")

        if len(self.warnings) > 5:
            logger.warning(f"  ... and {len(self.warnings) - 5} more warnings")


class DataValidator:
    """
    Comprehensive data validator for FQDN datasets.

    Validates:
    - Required columns existence
    - Data types
    - Missing values
    - Value ranges
    - Business rules (FQDN format, etc.)
    """

    def __init__(
        self,
        required_columns: list[str | None] = None,
        schema: dict[str, dict[str, Any | None]] = None,
        max_missing_ratio: float = 0.5,
        max_duplicate_ratio: float = 0.1,
    ):
        """
        Initialize data validator.

        Args:
            required_columns: List of required column names
            schema: Schema definition for columns
            max_missing_ratio: Maximum allowed missing value ratio
            max_duplicate_ratio: Maximum allowed duplicate ratio
        """
        self.required_columns = required_columns or [
            "fqdn",
            "record_type",
            "record_data",
            "appid",
        ]
        self.schema = schema or {}
        self.max_missing_ratio = max_missing_ratio
        self.max_duplicate_ratio = max_duplicate_ratio

        # FQDN validation pattern
        self.fqdn_pattern = re.compile(
            r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\.?$"
        )

    def validate(self, df: pl.DataFrame) -> ValidationResult:
        """
        Run all validation checks on DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with issues and statistics
        """
        result = ValidationResult(is_valid=True)

        # Basic statistics
        result.statistics = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns,
        }

        # Run validation checks
        self._validate_required_columns(df, result)
        self._validate_data_types(df, result)
        self._validate_missing_values(df, result)
        self._validate_duplicates(df, result)
        self._validate_fqdn_format(df, result)
        self._validate_target_column(df, result)
        self._validate_business_rules(df, result)

        # Add column statistics
        result.statistics["column_stats"] = self._compute_column_stats(df)

        result.log_summary()
        return result

    def _validate_required_columns(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check that required columns exist."""
        missing = set(self.required_columns) - set(df.columns)

        for col in missing:
            result.add_issue(
                column=col,
                issue_type="missing_column",
                message=f"Required column '{col}' is missing",
                severity=ValidationSeverity.ERROR,
            )

        result.statistics["required_columns"] = {
            "expected": self.required_columns,
            "missing": list(missing),
        }

    def _validate_data_types(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Validate column data types."""
        type_mapping = {
            "string": [pl.Utf8, pl.String],
            "integer": [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ],
            "float": [pl.Float32, pl.Float64],
            "boolean": [pl.Boolean],
            "datetime": [pl.Datetime, pl.Date],
        }

        for col, spec in self.schema.items():
            if col not in df.columns:
                continue

            expected_type = spec.get("type")
            if expected_type is None:
                continue

            actual_type = df[col].dtype
            valid_types = type_mapping.get(expected_type, [])

            if valid_types and actual_type not in valid_types:
                result.add_issue(
                    column=col,
                    issue_type="invalid_type",
                    message=f"Column '{col}' has type {actual_type}, expected {expected_type}",
                    severity=ValidationSeverity.WARNING,
                    details={"expected": expected_type, "actual": str(actual_type)},
                )

    def _validate_missing_values(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for excessive missing values."""
        n_rows = len(df)

        for col in df.columns:
            null_count = df[col].null_count()
            null_ratio = null_count / n_rows if n_rows > 0 else 0

            # Check schema for nullable
            spec = self.schema.get(col, {})
            nullable = spec.get("nullable", True)

            if not nullable and null_count > 0:
                result.add_issue(
                    column=col,
                    issue_type="null_in_non_nullable",
                    message=f"Column '{col}' has {null_count} null values but is not nullable",
                    severity=ValidationSeverity.ERROR,
                    details={"null_count": null_count, "null_ratio": null_ratio},
                )
            elif null_ratio > self.max_missing_ratio:
                result.add_issue(
                    column=col,
                    issue_type="high_missing_ratio",
                    message=f"Column '{col}' has {null_ratio:.1%} missing values",
                    severity=ValidationSeverity.WARNING,
                    details={"null_count": null_count, "null_ratio": null_ratio},
                )

    def _validate_duplicates(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for duplicate records."""
        # Check for exact duplicates
        n_rows = len(df)
        n_unique = df.n_unique()
        duplicate_ratio = 1 - (n_unique / n_rows) if n_rows > 0 else 0

        result.statistics["duplicates"] = {
            "total_rows": n_rows,
            "unique_rows": n_unique,
            "duplicate_ratio": duplicate_ratio,
        }

        if duplicate_ratio > self.max_duplicate_ratio:
            result.add_issue(
                column=None,
                issue_type="high_duplicate_ratio",
                message=f"Dataset has {duplicate_ratio:.1%} duplicate rows",
                severity=ValidationSeverity.WARNING,
                details={"duplicate_ratio": duplicate_ratio},
            )

        # Check for duplicate FQDNs
        if "fqdn" in df.columns:
            fqdn_counts = df.group_by("fqdn").count()
            duplicates = fqdn_counts.filter(pl.col("count") > 1)

            if len(duplicates) > 0:
                result.add_issue(
                    column="fqdn",
                    issue_type="duplicate_fqdns",
                    message=f"Found {len(duplicates)} FQDNs with multiple records",
                    severity=ValidationSeverity.INFO,
                    details={"duplicate_fqdn_count": len(duplicates)},
                )

    def _validate_fqdn_format(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Validate FQDN format."""
        if "fqdn" not in df.columns:
            return

        # Sample FQDNs for validation (checking all would be slow)
        sample_size = min(1000, len(df))
        sample = df.select("fqdn").sample(n=sample_size, seed=42)

        invalid_count = 0
        invalid_examples = []

        for fqdn in sample["fqdn"].to_list():
            if fqdn is None:
                continue

            # Basic format validation
            if not self._is_valid_fqdn(fqdn):
                invalid_count += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(fqdn)

        if invalid_count > 0:
            estimated_invalid = int(invalid_count * len(df) / sample_size)
            result.add_issue(
                column="fqdn",
                issue_type="invalid_fqdn_format",
                message=f"Approximately {estimated_invalid} FQDNs have invalid format",
                severity=ValidationSeverity.WARNING,
                details={
                    "sample_invalid": invalid_count,
                    "examples": invalid_examples,
                },
            )

    def _is_valid_fqdn(self, fqdn: str) -> bool:
        """Check if FQDN has valid format."""
        if not fqdn or len(fqdn) > 253:
            return False

        # Allow localhost and internal patterns
        if fqdn.startswith("localhost") or fqdn.endswith(".local"):
            return True

        # Check basic pattern
        if not self.fqdn_pattern.match(fqdn):
            # Allow some flexibility for internal domains
            parts = fqdn.split(".")
            if len(parts) >= 2:
                return all(
                    len(p) <= 63 and p and not p.startswith("-") and not p.endswith("-")
                    for p in parts
                )
            return False

        return True

    def _validate_target_column(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Validate target column (appid)."""
        target_col = "appid"

        if target_col not in df.columns:
            return

        # Check for negative values
        negative_count = df.filter(pl.col(target_col) < 0).height
        if negative_count > 0:
            result.add_issue(
                column=target_col,
                issue_type="negative_appid",
                message=f"Found {negative_count} negative appId values",
                severity=ValidationSeverity.WARNING,
            )

        # Class distribution
        class_counts = df.group_by(target_col).count().sort("count", descending=True)

        n_classes = len(class_counts)
        orphan_count = df.filter(pl.col(target_col) == 0).height
        labeled_count = len(df) - orphan_count

        result.statistics["target"] = {
            "column": target_col,
            "n_classes": n_classes,
            "orphan_count": orphan_count,
            "labeled_count": labeled_count,
            "orphan_ratio": orphan_count / len(df) if len(df) > 0 else 0,
        }

        # Check for rare classes
        rare_threshold = 3
        rare_classes = class_counts.filter(
            (pl.col("count") < rare_threshold) & (pl.col(target_col) != 0)
        )

        if len(rare_classes) > 0:
            result.add_issue(
                column=target_col,
                issue_type="rare_classes",
                message=f"Found {len(rare_classes)} classes with fewer than {rare_threshold} samples",
                severity=ValidationSeverity.INFO,
                details={"rare_class_count": len(rare_classes)},
            )

    def _validate_business_rules(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Validate business-specific rules."""
        # Check record_type values
        if "record_type" in df.columns:
            valid_types = {"A", "AAAA", "CNAME", "MX", "TXT", "NS", "PTR", "SRV", "SOA"}
            actual_types = set(df["record_type"].drop_nulls().unique().to_list())
            unknown_types = actual_types - valid_types

            if unknown_types:
                result.add_issue(
                    column="record_type",
                    issue_type="unknown_record_types",
                    message=f"Found unknown record types: {unknown_types}",
                    severity=ValidationSeverity.INFO,
                    details={"unknown_types": list(unknown_types)},
                )

        # Check fqdn_status values
        if "fqdn_status" in df.columns:
            status_counts = df.group_by("fqdn_status").count()
            result.statistics["status_distribution"] = {
                row["fqdn_status"]: row["count"] for row in status_counts.to_dicts()
            }

    def _compute_column_stats(
        self,
        df: pl.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Compute statistics for each column."""
        stats = {}

        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].null_count(),
                "null_ratio": df[col].null_count() / len(df) if len(df) > 0 else 0,
            }

            # Add type-specific stats
            dtype = df[col].dtype

            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                col_stats.update(
                    {
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                    }
                )
            elif dtype in [pl.Utf8, pl.String]:
                non_null = df[col].drop_nulls()
                col_stats.update(
                    {
                        "n_unique": df[col].n_unique(),
                        "min_length": (
                            non_null.str.len_chars().min()
                            if len(non_null) > 0
                            else None
                        ),
                        "max_length": (
                            non_null.str.len_chars().max()
                            if len(non_null) > 0
                            else None
                        ),
                    }
                )

            stats[col] = col_stats

        return stats


def validate_data(
    df: pl.DataFrame,
    required_columns: list[str | None] = None,
    schema: dict[str, dict[str, Any | None]] = None,
    max_missing_ratio: float = 0.5,
) -> ValidationResult:
    """
    Convenience function to validate a DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: Required column names
        schema: Column schema definitions
        max_missing_ratio: Maximum missing value ratio

    Returns:
        ValidationResult
    """
    validator = DataValidator(
        required_columns=required_columns,
        schema=schema,
        max_missing_ratio=max_missing_ratio,
    )
    return validator.validate(df)
