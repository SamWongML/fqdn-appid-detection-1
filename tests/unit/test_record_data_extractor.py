"""Unit tests for record data feature extractor."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.record_data_extractor import RecordDataExtractor


class TestRecordDataExtractor:
    """Tests for RecordDataExtractor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with various record types."""
        return pl.DataFrame(
            {
                "fqdn": [
                    "api.example.hsbc.com",
                    "www.hsbc.de",
                    "mail.hsbcnet.com",
                    "_dmarc.hsbc.co.uk",
                    "internal.corp.hsbc.com",
                ],
                "data": [
                    "api.gslb-uk1.hsbc.com",  # GSLB pattern
                    "frankfurt-alb-123.elb.eu-west-1.amazonaws.com",  # AWS ALB
                    "10.0.0.1",  # Private IP
                    "v=DMARC1; p=reject; fo=1;",  # DMARC record
                    "192.168.1.100",  # Private IP
                ],
                "record_type": ["CNAME", "CNAME", "A", "TXT", "A"],
            }
        )

    @pytest.fixture
    def extractor(self):
        """Create an extractor instance."""
        return RecordDataExtractor()

    def test_extract_cloud_provider_aws(self, sample_data, extractor):
        """Test AWS cloud provider detection."""
        result = extractor.extract_features(sample_data)

        assert "data_is_aws" in result.columns
        # Second row has AWS ELB pattern
        assert result["data_is_aws"][1] == 1
        # First row is not AWS
        assert result["data_is_aws"][0] == 0

    def test_extract_gslb_pattern(self, sample_data, extractor):
        """Test GSLB pattern detection."""
        result = extractor.extract_features(sample_data)

        assert "data_has_gslb" in result.columns
        # First row has GSLB pattern
        assert result["data_has_gslb"][0] == 1
        # Other rows don't have GSLB
        assert result["data_has_gslb"][1] == 0

    def test_extract_private_ip(self, sample_data, extractor):
        """Test private IP detection for A records."""
        result = extractor.extract_features(sample_data)

        assert "data_is_private_ip" in result.columns
        # Third row (10.0.0.1) is private IP
        assert result["data_is_private_ip"][2] == 1
        # Fifth row (192.168.x) is private IP
        assert result["data_is_private_ip"][4] == 1
        # First row (CNAME) should be 0
        assert result["data_is_private_ip"][0] == 0

    def test_extract_dmarc_from_txt(self, sample_data, extractor):
        """Test DMARC detection from TXT records."""
        result = extractor.extract_features(sample_data)

        assert "data_is_dmarc" in result.columns
        # Fourth row is DMARC TXT record
        assert result["data_is_dmarc"][3] == 1
        # Other rows should be 0
        assert result["data_is_dmarc"][0] == 0

    def test_structural_features(self, sample_data, extractor):
        """Test structural feature extraction."""
        result = extractor.extract_features(sample_data)

        assert "data_subdomain_depth" in result.columns
        assert "data_length" in result.columns

        # "api.gslb-uk1.hsbc.com" has 4 parts
        assert result["data_subdomain_depth"][0] == 4

    def test_feature_names_property(self, extractor):
        """Test that feature_names property returns expected features."""
        names = extractor.feature_names

        assert "data_is_aws" in names
        assert "data_is_azure" in names
        assert "data_has_gslb" in names
        assert "data_is_private_ip" in names
        assert "data_subdomain_depth" in names

    def test_missing_data_column(self, extractor):
        """Test handling of missing data column."""
        df = pl.DataFrame({"fqdn": ["test.com"], "record_type": ["A"]})
        result = extractor.extract_features(df)

        # Should return original dataframe without error
        assert len(result) == 1

    def test_null_values_handling(self, extractor):
        """Test handling of null values in data column."""
        df = pl.DataFrame(
            {
                "fqdn": ["test.com", "test2.com"],
                "data": [None, "valid.example.com"],
                "record_type": ["A", "CNAME"],
            }
        )
        result = extractor.extract_features(df)

        # Should not raise exception
        assert len(result) == 2
        # Null should result in 0 for cloud detection
        assert result["data_is_aws"][0] == 0


class TestAppDescriptionVectorizer:
    """Tests for AppDescriptionVectorizer."""

    @pytest.fixture
    def sample_descriptions(self):
        """Create sample app descriptions."""
        return [
            "Customer portal for online banking and payment services",
            "Internal email gateway for security compliance",
            "Trading platform for corporate wealth management",
            "Mobile API for retail banking customers",
            "",  # Empty description
        ]

    def test_fit_transform(self):
        """Test fit and transform of app descriptions."""
        from src.features.feature_engineer import AppDescriptionVectorizer

        descriptions = [
            "Customer portal for banking services",
            "Email gateway security service",
            "Trading platform for wealth management",
            "Mobile API for retail banking",
            "Internal compliance security portal",
        ]

        # Use min_df=1 for small test datasets
        vectorizer = AppDescriptionVectorizer(max_features=50, min_df=1)
        result = vectorizer.fit_transform(descriptions, column_name="appdesc")

        assert result.shape[0] == 5
        assert result.shape[1] <= 50

    def test_keyword_features(self):
        """Test binary keyword feature extraction."""
        from src.features.feature_engineer import AppDescriptionVectorizer

        descriptions = [
            "Customer payment portal",  # Has 'customer', 'payment', 'portal'
            "Internal API service",  # Has 'api', 'internal'
            "Simple test app",  # No keywords
        ]

        vectorizer = AppDescriptionVectorizer()
        keywords = vectorizer.extract_keyword_features(descriptions)

        assert keywords.shape[0] == 3
        assert keywords.shape[1] == len(vectorizer.BUSINESS_KEYWORDS)

        # First row should have 'customer', 'payment', 'portal' set
        customer_idx = vectorizer.BUSINESS_KEYWORDS.index("customer")
        assert keywords[0, customer_idx] == 1

    def test_preprocessing(self):
        """Test text preprocessing."""
        from src.features.feature_engineer import AppDescriptionVectorizer

        vectorizer = AppDescriptionVectorizer()

        # Test CR removal
        result = vectorizer._preprocess_text("Hello<CR>World")
        assert "<CR>" not in result

        # Test lowercasing
        result = vectorizer._preprocess_text("HELLO WORLD")
        assert result == "hello world"


class TestScanResultExtractor:
    """Tests for ScanResultExtractor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with scan results."""
        return pl.DataFrame(
            {
                "fqdn": ["test1.com", "test2.com", "test3.com"],
                "scan_result": [
                    ["cname_ip_present"],
                    ["rxdomain", "no_host_records"],
                    None,
                ],
            }
        )

    def test_extract_scan_features(self):
        """Test scan result feature extraction."""
        from src.features.feature_engineer import ScanResultExtractor

        df = pl.DataFrame(
            {
                "fqdn": ["test1.com", "test2.com"],
                "scan_result": [
                    ["cname_ip_present"],
                    ["rxdomain", "no_host_records"],
                ],
            }
        )

        extractor = ScanResultExtractor()
        result = extractor.extract_features(df)

        assert "scan_cname_ip_present" in result.columns
        assert "scan_rxdomain" in result.columns
        assert "scan_count" in result.columns

        # First row has cname_ip_present
        assert result["scan_cname_ip_present"][0] == 1
        assert result["scan_rxdomain"][0] == 0

        # Second row has rxdomain and no_host_records
        assert result["scan_rxdomain"][1] == 1
        assert result["scan_no_host_records"][1] == 1
        assert result["scan_count"][1] == 2

    def test_null_scan_result(self):
        """Test handling of null scan results."""
        from src.features.feature_engineer import ScanResultExtractor

        df = pl.DataFrame(
            {
                "fqdn": ["test.com"],
                "scan_result": [None],
            }
        )

        extractor = ScanResultExtractor()
        result = extractor.extract_features(df)

        # All scan features should be 0
        assert result["scan_cname_ip_present"][0] == 0
        assert result["scan_count"][0] == 0

    def test_missing_scan_result_column(self):
        """Test handling when scan_result column is missing."""
        from src.features.feature_engineer import ScanResultExtractor

        df = pl.DataFrame({"fqdn": ["test.com"]})

        extractor = ScanResultExtractor()
        result = extractor.extract_features(df)

        # Should add zero-filled columns
        assert "scan_cname_ip_present" in result.columns
        assert result["scan_cname_ip_present"][0] == 0
