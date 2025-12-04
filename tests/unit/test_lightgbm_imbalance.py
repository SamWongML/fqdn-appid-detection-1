"""Unit tests for LightGBM imbalance handling."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import LightGBMSettings
from src.models.base_model import LightGBMModel


class TestLightGBMImbalance:
    """Tests for LightGBM imbalance parameters."""

    @pytest.fixture
    def train_data(self):
        """Generate training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        return X, y

    def test_class_weight_parameter(self, train_data):
        """Test that class_weight is correctly passed to the model."""
        X_train, y_train = train_data

        settings = LightGBMSettings(class_weight="balanced")
        model = LightGBMModel(settings=settings)

        # Check if parameter is in _params
        assert model._params["class_weight"] == "balanced"

        # Fit model to ensure no errors
        model.fit(X_train, y_train)
        assert model.is_fitted

    def test_is_unbalance_parameter(self, train_data):
        """Test that is_unbalance is correctly passed to the model."""
        X_train, y_train = train_data

        settings = LightGBMSettings(is_unbalance=True)
        model = LightGBMModel(settings=settings)

        # Check if parameter is in _params
        assert model._params["is_unbalance"] is True

        # Fit model to ensure no errors
        model.fit(X_train, y_train)
        assert model.is_fitted

    def test_default_parameters(self):
        """Test default values for imbalance parameters."""
        settings = LightGBMSettings()
        model = LightGBMModel(settings=settings)

        assert model._params["class_weight"] is None
        assert model._params["is_unbalance"] is False
