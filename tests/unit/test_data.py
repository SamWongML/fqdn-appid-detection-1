"""Unit tests for data loading and preprocessing modules."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.data_loader import CSVDataLoader, create_data_loader
from src.data.data_validator import DataValidator, ValidationResult
from src.data.preprocessor import DataPreprocessor


class TestCSVDataLoader:
    """Tests for CSVDataLoader."""

    def test_load_from_csv(self, temp_csv_file):
        """Test loading data from CSV file."""
        loader = CSVDataLoader(labeled_path=temp_csv_file)
        df = loader.load_labeled()

        assert len(df) > 0
        assert "fqdn" in df.columns
        assert "appid" in df.columns

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file."""
        loader = CSVDataLoader(labeled_path=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_labeled()

    def test_split_data(self, sample_labeled_df, tmp_path):
        """Test data splitting."""
        csv_path = tmp_path / "data.csv"
        sample_labeled_df.write_csv(csv_path)

        loader = CSVDataLoader(labeled_path=csv_path)
        df = loader.load_labeled()

        train, val, test = loader.split_data(df)

        # Check splits sum to total
        assert len(train) + len(val) + len(test) == len(df)

        # Check no overlap
        train_fqdns = set(train["fqdn"].to_list())
        val_fqdns = set(val["fqdn"].to_list())
        test_fqdns = set(test["fqdn"].to_list())

        assert len(train_fqdns & val_fqdns) == 0
        assert len(train_fqdns & test_fqdns) == 0
        assert len(val_fqdns & test_fqdns) == 0


class TestDataValidator:
    """Tests for DataValidator."""

    def test_valid_data(self, sample_labeled_df):
        """Test validation of valid data."""
        validator = DataValidator()
        result = validator.validate(sample_labeled_df)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        df = pl.DataFrame({"other_column": ["value"]})

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_invalid_fqdn_format(self):
        """Test validation with invalid FQDN format."""
        df = pl.DataFrame(
            {
                "fqdn": ["valid.com", "invalid..fqdn", "-invalid.com"],
                "record_type": ["A", "A", "A"],
                "record_data": ["1.1.1.1", "2.2.2.2", "3.3.3.3"],
                "appid": [1, 2, 3],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)

        # Should have warnings about invalid FQDNs
        assert len(result.warnings) > 0 or len(result.errors) > 0

    def test_validation_statistics(self, sample_labeled_df):
        """Test that validation computes statistics."""
        validator = DataValidator()
        result = validator.validate(sample_labeled_df)

        assert "statistics" in dir(result) or result.statistics is not None


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    def test_fit_transform(self, sample_labeled_df):
        """Test preprocessing fit and transform."""
        preprocessor = DataPreprocessor(target_column="appid")
        result = preprocessor.fit_transform(sample_labeled_df)

        assert len(result) <= len(sample_labeled_df)
        assert "appid" in result.columns

    def test_class_mapping(self, sample_labeled_df):
        """Test class mapping is created."""
        preprocessor = DataPreprocessor(
            target_column="appid", min_samples_per_class=1, handle_rare_classes="keep"
        )
        preprocessor.fit_transform(sample_labeled_df)

        assert preprocessor.class_mapping is not None

        # Test mapping
        labels = sample_labeled_df["appid"].unique().to_list()
        for label in labels:
            if label != 0:  # Skip orphan value
                idx = preprocessor.class_mapping.label_to_idx(label)
                assert preprocessor.class_mapping.idx_to_label(idx) == label

    def test_handle_rare_classes_drop(self):
        """Test dropping rare classes."""
        df = pl.DataFrame(
            {
                "fqdn": ["a.com", "b.com", "c.com", "d.com", "e.com"],
                "record_type": ["A"] * 5,
                "record_data": ["1.1.1.1"] * 5,
                "appid": [1, 1, 1, 2, 3],  # Class 2 and 3 are rare
            }
        )

        preprocessor = DataPreprocessor(
            target_column="appid",
            min_samples_per_class=2,
            handle_rare_classes="drop",
        )
        result = preprocessor.fit_transform(df)

        # Rare classes should be dropped
        remaining_classes = result["appid"].unique().to_list()
        assert 2 not in remaining_classes
        assert 3 not in remaining_classes

    def test_normalize_text(self, sample_labeled_df):
        """Test text normalization."""
        # Add mixed case FQDNs
        df = sample_labeled_df.with_columns(
            pl.col("fqdn").str.to_uppercase().alias("fqdn")
        )

        preprocessor = DataPreprocessor(target_column="appid")
        result = preprocessor.fit_transform(df)

        # FQDNs should be normalized (lowercase)
        for fqdn in result["fqdn"].to_list():
            assert fqdn == fqdn.lower()

    def test_save_load(self, sample_labeled_df, tmp_path):
        """Test saving and loading preprocessor."""
        preprocessor = DataPreprocessor(target_column="appid")
        preprocessor.fit_transform(sample_labeled_df)

        # Save
        save_path = tmp_path / "preprocessor.joblib"
        preprocessor.save(str(save_path))

        # Load
        loaded = DataPreprocessor.load(str(save_path))

        # Transform should produce same results
        # Note: fit_transform creates class mapping, so we need to compare transform
        result1 = preprocessor.transform(sample_labeled_df)
        result2 = loaded.transform(sample_labeled_df)

        assert len(result1) == len(result2)
