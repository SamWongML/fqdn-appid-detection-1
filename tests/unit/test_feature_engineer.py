"""Unit tests for feature engineering module."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.feature_engineer import (
    CategoricalEncoder,
    FeatureEngineer,
    FQDNFeatureExtractor,
    TextVectorizer,
)


class TestFQDNFeatureExtractor:
    """Tests for FQDNFeatureExtractor."""

    def test_extract_structural_features(self, sample_dataframe):
        """Test structural feature extraction."""
        extractor = FQDNFeatureExtractor()
        result = extractor.extract_features(sample_dataframe)

        # Check new columns exist
        assert "fqdn_length" in result.columns
        assert "fqdn_num_dots" in result.columns
        assert "fqdn_num_hyphens" in result.columns
        assert "fqdn_num_labels" in result.columns

    def test_fqdn_length(self, sample_dataframe):
        """Test FQDN length calculation."""
        extractor = FQDNFeatureExtractor()
        result = extractor.extract_features(sample_dataframe)

        # Check length of first FQDN: "api.dev.example.com" = 19
        assert result["fqdn_length"][0] == 19

    def test_dot_count(self, sample_dataframe):
        """Test dot counting."""
        extractor = FQDNFeatureExtractor()
        result = extractor.extract_features(sample_dataframe)

        # "api.dev.example.com" has 3 dots
        assert result["fqdn_num_dots"][0] == 3

    def test_pattern_extraction(self, sample_dataframe):
        """Test pattern-based features."""
        extractor = FQDNFeatureExtractor()
        result = extractor.extract_features(sample_dataframe)

        # Check dev pattern detected
        assert "fqdn_is_dev" in result.columns
        # First FQDN has "dev" in it
        assert result["fqdn_is_dev"][0] == 1

    def test_empty_fqdn(self):
        """Test handling of empty FQDNs."""
        df = pl.DataFrame({"fqdn": ["", None, "valid.example.com"]})
        extractor = FQDNFeatureExtractor()
        result = extractor.extract_features(df)

        # Should not raise exception
        assert len(result) == 3


class TestTextVectorizer:
    """Tests for TextVectorizer."""

    def test_fit_transform(self, sample_fqdns):
        """Test fit and transform."""
        vectorizer = TextVectorizer(max_features=100)
        result = vectorizer.fit_transform(sample_fqdns, column_name="fqdn")

        assert result.shape[0] == len(sample_fqdns)
        assert result.shape[1] <= 100

    def test_feature_names(self, sample_fqdns):
        """Test feature name generation."""
        vectorizer = TextVectorizer(max_features=50)
        vectorizer.fit(sample_fqdns, column_name="fqdn")

        assert len(vectorizer.feature_names) > 0
        assert all("fqdn_tfidf_" in name for name in vectorizer.feature_names)

    def test_transform_without_fit(self):
        """Test error when transforming without fit."""
        vectorizer = TextVectorizer()
        with pytest.raises(RuntimeError):
            vectorizer.transform(["test.com"])


class TestCategoricalEncoder:
    """Tests for CategoricalEncoder."""

    def test_label_encoding(self, sample_dataframe):
        """Test label encoding strategy."""
        encoder = CategoricalEncoder(strategy="label")
        encoder.fit(sample_dataframe, columns=["record_type"])

        result = encoder.transform(sample_dataframe, columns=["record_type"])

        assert result.shape[0] == len(sample_dataframe)
        assert result.shape[1] == 1

    def test_frequency_encoding(self, sample_dataframe):
        """Test frequency encoding."""
        encoder = CategoricalEncoder(strategy="frequency")
        encoder.fit(sample_dataframe, columns=["brand"])

        result = encoder.transform(sample_dataframe, columns=["brand"])

        # Frequencies should be between 0 and 1
        assert result.min() >= 0
        assert result.max() <= 1

    def test_unknown_category(self, sample_dataframe):
        """Test handling of unknown categories."""
        encoder = CategoricalEncoder(strategy="label")
        encoder.fit(sample_dataframe, columns=["record_type"])

        # Create test data with unknown category
        test_df = pl.DataFrame({"record_type": ["UNKNOWN_TYPE"]})
        result = encoder.transform(test_df, columns=["record_type"])

        # Should return default value (not crash)
        assert result.shape[0] == 1


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_fit_transform(self, sample_dataframe):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer(
            fqdn_column="fqdn",
            categorical_columns=["record_type"],
            tfidf_max_features=50,
        )

        X, feature_names = engineer.fit_transform(sample_dataframe)

        assert X.shape[0] == len(sample_dataframe)
        assert X.shape[1] == len(feature_names)
        assert X.shape[1] > 0

    def test_transform_after_fit(self, sample_dataframe):
        """Test transform after fitting."""
        engineer = FeatureEngineer(
            fqdn_column="fqdn",
            categorical_columns=["record_type"],
            tfidf_max_features=50,
        )
        engineer.fit(sample_dataframe)
        X, _ = engineer.transform(sample_dataframe)

        assert X.shape[0] == len(sample_dataframe)

    def test_feature_count(self, sample_dataframe):
        """Test feature count property."""
        engineer = FeatureEngineer(
            fqdn_column="fqdn",
            categorical_columns=["record_type"],
            tfidf_max_features=50,
        )
        engineer.fit(sample_dataframe)

        assert engineer.n_features > 0
        assert engineer.n_features == len(engineer.feature_names)

    def test_no_nan_values(self, sample_dataframe):
        """Test that output contains no NaNs."""
        engineer = FeatureEngineer(
            fqdn_column="fqdn",
            categorical_columns=["record_type"],
            tfidf_max_features=50,
        )
        X, _ = engineer.fit_transform(sample_dataframe)

        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
