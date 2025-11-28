"""Unit tests for feature engineering module."""

import numpy as np
import polars as pl
import pytest

from src.features.feature_engineer import (
    CategoricalEncoder,
    FeatureEngineer,
    FQDNFeatureExtractor,
    TextVectorizer,
)


class TestFQDNFeatureExtractor:
    """Tests for FQDNFeatureExtractor."""

    def test_extract_structural_features(self, sample_fqdns):
        """Test structural feature extraction."""
        extractor = FQDNFeatureExtractor()
        df = pl.DataFrame({"fqdn": sample_fqdns})

        result = extractor.extract_features(df)

        # Check structural features exist
        assert "fqdn_length" in result.columns
        assert "fqdn_num_dots" in result.columns
        assert "fqdn_num_hyphens" in result.columns
        assert "fqdn_num_labels" in result.columns

        # Verify values
        assert result["fqdn_length"][0] == 19  # len("api.dev.example.com")
        assert result["fqdn_num_dots"][0] == 3

    def test_extract_pattern_features(self, sample_fqdns):
        """Test pattern feature extraction."""
        extractor = FQDNFeatureExtractor()
        df = pl.DataFrame({"fqdn": sample_fqdns})

        result = extractor.extract_features(df)

        # Check pattern features exist
        assert "fqdn_is_api" in result.columns
        assert "fqdn_is_dev" in result.columns
        assert "fqdn_is_cdn" in result.columns

        # Verify patterns detected
        api_row = result.filter(pl.col("fqdn").str.contains("api"))
        assert api_row["fqdn_is_api"][0] == 1

        dev_row = result.filter(pl.col("fqdn").str.contains("dev"))
        assert dev_row["fqdn_is_dev"][0] == 1

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        extractor = FQDNFeatureExtractor()
        df = pl.DataFrame({"fqdn": []}, schema={"fqdn": pl.String})

        result = extractor.extract_features(df)
        assert len(result) == 0

    def test_missing_fqdn_column(self):
        """Test with missing FQDN column."""
        extractor = FQDNFeatureExtractor()
        df = pl.DataFrame({"other_column": ["value"]})

        result = extractor.extract_features(df)
        # Should return original DataFrame unchanged
        assert "fqdn_length" not in result.columns


class TestTextVectorizer:
    """Tests for TextVectorizer."""

    def test_fit_transform(self, sample_fqdns):
        """Test fit and transform."""
        vectorizer = TextVectorizer(max_features=100)

        result = vectorizer.fit_transform(sample_fqdns, column_name="fqdn")

        assert result.shape[0] == len(sample_fqdns)
        assert result.shape[1] <= 100
        assert len(vectorizer.feature_names) == result.shape[1]

    def test_transform_without_fit(self):
        """Test transform without fitting raises error."""
        vectorizer = TextVectorizer()

        with pytest.raises(RuntimeError, match="not fitted"):
            vectorizer.transform(["test.com"])

    def test_feature_names(self, sample_fqdns):
        """Test feature names are generated correctly."""
        vectorizer = TextVectorizer(max_features=50)
        vectorizer.fit(sample_fqdns, column_name="test")

        for name in vectorizer.feature_names:
            assert name.startswith("test_tfidf_")


class TestCategoricalEncoder:
    """Tests for CategoricalEncoder."""

    def test_label_encoding(self, sample_labeled_df):
        """Test label encoding."""
        encoder = CategoricalEncoder(strategy="label")
        encoder.fit(sample_labeled_df, columns=["record_type"])

        result = encoder.transform(sample_labeled_df, columns=["record_type"])

        assert result.shape[0] == len(sample_labeled_df)
        assert result.shape[1] == 1
        # Values should be integers
        assert result.dtype == np.float64 or np.issubdtype(result.dtype, np.integer)

    def test_frequency_encoding(self, sample_labeled_df):
        """Test frequency encoding."""
        encoder = CategoricalEncoder(strategy="frequency")
        encoder.fit(sample_labeled_df, columns=["record_type"])

        result = encoder.transform(sample_labeled_df, columns=["record_type"])

        # Frequencies should be between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_unknown_category_handling(self, sample_labeled_df):
        """Test handling of unknown categories."""
        encoder = CategoricalEncoder(strategy="label")
        encoder.fit(sample_labeled_df, columns=["record_type"])

        # Create data with unknown category
        new_df = pl.DataFrame({"record_type": ["UNKNOWN_TYPE"]})
        result = encoder.transform(new_df, columns=["record_type"])

        # Should use default value
        assert result.shape == (1, 1)


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_fit_transform(self, sample_labeled_df):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer(
            fqdn_column="fqdn",
            categorical_columns=["record_type"],
            tfidf_max_features=50,
        )

        X, feature_names = engineer.fit_transform(sample_labeled_df)

        assert X.shape[0] == len(sample_labeled_df)
        assert X.shape[1] == len(feature_names)
        assert X.shape[1] > 0

    def test_transform_without_fit(self, sample_labeled_df):
        """Test transform without fitting raises error."""
        engineer = FeatureEngineer()

        with pytest.raises(RuntimeError, match="not fitted"):
            engineer.transform(sample_labeled_df)

    def test_save_load(self, sample_labeled_df, tmp_path):
        """Test saving and loading feature engineer."""
        engineer = FeatureEngineer(tfidf_max_features=30)
        engineer.fit(sample_labeled_df)

        # Save
        save_path = tmp_path / "feature_engineer.joblib"
        engineer.save(str(save_path))

        # Load
        loaded = FeatureEngineer.load(str(save_path))

        # Compare transforms
        X1, _ = engineer.transform(sample_labeled_df)
        X2, _ = loaded.transform(sample_labeled_df)

        np.testing.assert_array_almost_equal(X1, X2)
