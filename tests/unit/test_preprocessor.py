"""Unit tests for data preprocessing module."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with various data quality issues."""
        return pl.DataFrame(
            {
                "fqdn": [
                    "Api.Example.COM.",
                    "  www.test.net  ",
                    "service.INTERNAL.org",
                    None,
                    "valid.domain.io",
                ],
                "record_type": ["A", "CNAME", None, "A", "A"],
                "record_data": [
                    "192.168.1.1",
                    None,
                    "10.0.0.1",
                    "172.16.0.1",
                    "8.8.8.8",
                ],
                "appid": [1001, 1002, 1001, 1003, 1001],
                "brand": ["BrandA", "BrandB", None, "BrandA", "BrandC"],
            }
        )

    def test_normalize_text(self, sample_df):
        """Test text normalization (lowercase, strip, remove trailing dot)."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_df)

        # Check normalization
        fqdns = result["fqdn"].to_list()
        assert fqdns[0] == "api.example.com"  # lowercase, no trailing dot
        assert fqdns[1] == "www.test.net"  # stripped
        assert fqdns[2] == "service.internal.org"  # lowercase

    def test_handle_missing_values(self, sample_df):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_df)

        # Check no nulls in key columns
        assert result["record_type"].null_count() == 0
        assert result["brand"].null_count() == 0

    def test_rare_class_grouping(self):
        """Test rare class grouping."""
        df = pl.DataFrame(
            {
                "fqdn": [f"domain{i}.com" for i in range(100)],
                "appid": [1] * 50 + [2] * 30 + [3] * 10 + [4] * 5 + [5] * 3 + [6, 7],
            }
        )

        preprocessor = DataPreprocessor(
            handle_rare_classes="group",
            min_samples_per_class=5,
            rare_class_threshold=10,
        )
        result = preprocessor.fit_transform(df)

        # Classes with < 10 samples should be grouped
        unique_classes = result["appid"].unique().to_list()
        # 6 and 7 (single samples) and 5 (3 samples) should be grouped to rare_class_label
        assert -999 in unique_classes or len([c for c in unique_classes if c < 0]) > 0

    def test_rare_class_drop(self):
        """Test rare class dropping."""
        df = pl.DataFrame(
            {
                "fqdn": [f"domain{i}.com" for i in range(20)],
                "appid": [1] * 10 + [2] * 5 + [3] * 3 + [4, 5],
            }
        )

        preprocessor = DataPreprocessor(
            handle_rare_classes="drop",
            min_samples_per_class=3,
        )
        result = preprocessor.fit_transform(df)

        # Should have fewer records (classes with < 3 samples dropped)
        assert len(result) < 20

    def test_class_mapping(self, sample_df):
        """Test class label mapping."""
        preprocessor = DataPreprocessor(
            min_samples_per_class=1, handle_rare_classes="keep"
        )
        preprocessor.fit_transform(sample_df)

        mapping = preprocessor.class_mapping

        assert mapping is not None
        assert len(mapping.classes) > 0

        # Test round-trip
        original_label = 1001
        idx = mapping.label_to_idx(original_label)
        recovered = mapping.idx_to_label(idx)
        assert recovered == original_label

    def test_transform_new_data(self, sample_df):
        """Test transform on new data after fitting."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)

        new_df = pl.DataFrame(
            {
                "fqdn": ["NEW.DOMAIN.COM.", "another.test.net"],
                "record_type": ["A", "CNAME"],
                "record_data": ["1.2.3.4", "5.6.7.8"],
                "appid": [1001, 1002],
                "brand": ["BrandA", "BrandB"],
            }
        )

        result = preprocessor.transform(new_df)

        # Check normalization applied
        assert result["fqdn"][0] == "new.domain.com"

    def test_preserves_original_columns(self, sample_df):
        """Test that original columns are preserved."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_df)

        assert "fqdn" in result.columns
        assert "record_type" in result.columns
        assert "appid" in result.columns

    def test_save_load(self, sample_df, tmp_path):
        """Test preprocessor serialization."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)

        save_path = tmp_path / "preprocessor.joblib"
        preprocessor.save(str(save_path))

        assert save_path.exists()

        loaded = DataPreprocessor.load(str(save_path))

        # Should be able to transform with loaded preprocessor
        result = loaded.transform(sample_df)
        assert len(result) > 0


class TestDataValidator:
    """Tests for DataValidator (if implemented)."""

    def test_fqdn_format_validation(self):
        """Test FQDN format validation."""
        # This would test the validator if implemented
        pass

    def test_missing_column_detection(self):
        """Test detection of missing required columns."""
        pass
