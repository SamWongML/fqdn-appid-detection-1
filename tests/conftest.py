"""Pytest fixtures and configuration."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_fqdns():
    """Sample FQDN data."""
    return [
        "api.dev.example.com",
        "www.production.myapp.co.uk",
        "cdn.static.assets.net",
        "mail.corporate.internal.org",
        "db.mysql.staging.cloud",
    ]


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "fqdn": [
                "api.dev.example.com",
                "www.production.myapp.co.uk",
                "cdn.static.assets.net",
                "mail.corporate.internal.org",
                "db.mysql.staging.cloud",
                "test.qa.myservice.io",
                "admin.console.app.com",
                "auth.login.secure.net",
            ],
            "record_type": ["A", "CNAME", "A", "MX", "A", "A", "A", "CNAME"],
            "record_data": [
                "192.168.1.1",
                "prod.cloudfront.net",
                "10.0.0.1",
                "mail.example.com",
                "172.16.0.1",
                "192.168.2.1",
                "10.0.1.1",
                "auth.example.com",
            ],
            "appid": [1001, 1002, 1003, 1001, 1004, 1002, 1005, 1003],
            "brand": [
                "BrandA",
                "BrandB",
                "BrandA",
                "BrandC",
                "BrandB",
                "BrandA",
                "BrandC",
                "BrandB",
            ],
        }
    )


@pytest.fixture
def sample_labeled_df(sample_dataframe):
    """Alias for sample_dataframe (labeled data)."""
    return sample_dataframe


@pytest.fixture
def sample_orphan_dataframe():
    """Sample orphan DataFrame (appid = 0)."""
    return pl.DataFrame(
        {
            "fqdn": [
                "unknown.orphan.example.com",
                "mystery.service.net",
                "unlabeled.app.io",
            ],
            "record_type": ["A", "CNAME", "A"],
            "record_data": ["192.168.100.1", "mystery.cloudflare.net", "10.0.100.1"],
            "appid": [0, 0, 0],
        }
    )


@pytest.fixture
def sample_features():
    """Sample feature matrix."""
    np.random.seed(42)
    return np.random.randn(100, 50)


@pytest.fixture
def sample_labels():
    """Sample labels."""
    np.random.seed(42)
    return np.random.randint(0, 10, 100)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def temp_csv_file(tmp_path, sample_labeled_df):
    """Create a temporary CSV file with labeled data."""
    csv_path = tmp_path / "labeled_data.csv"
    sample_labeled_df.write_csv(csv_path)
    return csv_path
