"""Unit tests for models module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.base_model import (
    BaseModel,
    CatBoostModel,
    LightGBMModel,
    XGBoostModel,
    create_model,
)
from src.models.ensemble import EnsembleModel, create_ensemble


class TestBaseModels:
    """Tests for base model implementations."""

    @pytest.fixture
    def train_data(self):
        """Generate training data."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.random.randint(0, 5, 200)
        return X, y

    @pytest.fixture
    def val_data(self):
        """Generate validation data."""
        np.random.seed(43)
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 5, 50)
        return X, y

    def test_xgboost_fit_predict(self, train_data, val_data):
        """Test XGBoost model training and prediction."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        model = XGBoostModel()
        model.fit(X_train, y_train, X_val, y_val)

        assert model.is_fitted

        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)

        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), model.n_classes)

    def test_lightgbm_fit_predict(self, train_data, val_data):
        """Test LightGBM model training and prediction."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        model = LightGBMModel()
        model.fit(X_train, y_train, X_val, y_val)

        assert model.is_fitted

        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)

    def test_catboost_fit_predict(self, train_data, val_data):
        """Test CatBoost model training and prediction."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        model = CatBoostModel()
        model.fit(X_train, y_train, X_val, y_val)

        assert model.is_fitted

        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)

    def test_create_model_factory(self):
        """Test model factory function."""
        xgb = create_model("xgboost")
        assert isinstance(xgb, XGBoostModel)

        lgb = create_model("lightgbm")
        assert isinstance(lgb, LightGBMModel)

        cat = create_model("catboost")
        assert isinstance(cat, CatBoostModel)

    def test_create_model_invalid_type(self):
        """Test factory with invalid model type."""
        with pytest.raises(ValueError):
            create_model("invalid_model")

    def test_predict_without_fit(self, val_data):
        """Test prediction without fitting raises error."""
        X_val, _ = val_data
        model = XGBoostModel()

        with pytest.raises(RuntimeError):
            model.predict(X_val)

    def test_model_save_load(self, train_data, tmp_path):
        """Test model serialization."""
        X_train, y_train = train_data

        model = XGBoostModel()
        model.fit(X_train, y_train)

        # Save
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        assert save_path.exists()

        # Load
        loaded_model = XGBoostModel.load(save_path)

        assert loaded_model.is_fitted
        predictions = loaded_model.predict(X_train[:10])
        assert len(predictions) == 10

    def test_feature_importances(self, train_data):
        """Test feature importance extraction."""
        X_train, y_train = train_data

        model = XGBoostModel()
        model.fit(X_train, y_train)

        importances = model.feature_importances_
        assert len(importances) == X_train.shape[1]
        assert all(imp >= 0 for imp in importances)


class TestEnsembleModel:
    """Tests for ensemble model."""

    @pytest.fixture
    def train_data(self):
        """Generate training data."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.random.randint(0, 5, 200)
        return X, y

    @pytest.fixture
    def val_data(self):
        """Generate validation data."""
        np.random.seed(43)
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 5, 50)
        return X, y

    def test_ensemble_fit_predict(self, train_data, val_data):
        """Test ensemble training and prediction."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Create simple ensemble with just XGBoost and LightGBM
        models = [
            ("xgboost", XGBoostModel(), 0.5),
            ("lightgbm", LightGBMModel(), 0.5),
        ]
        ensemble = EnsembleModel(models=models)

        ensemble.fit(X_train, y_train, X_val, y_val)

        assert ensemble.is_fitted

        predictions = ensemble.predict(X_val)
        assert len(predictions) == len(X_val)

    def test_ensemble_soft_voting(self, train_data, val_data):
        """Test soft voting ensemble."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        models = [
            ("xgboost", XGBoostModel(), 0.6),
            ("lightgbm", LightGBMModel(), 0.4),
        ]
        ensemble = EnsembleModel(models=models, voting="soft")
        ensemble.fit(X_train, y_train, X_val, y_val)

        proba = ensemble.predict_proba(X_val)

        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_ensemble_with_confidence(self, train_data, val_data):
        """Test prediction with confidence scores."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        models = [
            ("xgboost", XGBoostModel(), 0.5),
            ("lightgbm", LightGBMModel(), 0.5),
        ]
        ensemble = EnsembleModel(models=models)
        ensemble.fit(X_train, y_train, X_val, y_val)

        predictions, confidences, top_k = ensemble.predict_with_confidence(
            X_val, top_k=3
        )

        assert len(predictions) == len(X_val)
        assert len(confidences) == len(X_val)
        assert top_k.shape == (len(X_val), 3)

        # Confidences should be between 0 and 1
        assert all(0 <= c <= 1 for c in confidences)

    def test_weight_normalization(self):
        """Test that weights are normalized."""
        models = [
            ("m1", XGBoostModel(), 2.0),
            ("m2", LightGBMModel(), 3.0),
        ]
        ensemble = EnsembleModel(models=models)

        weights = [w for _, _, w in ensemble.models]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_create_ensemble_factory(self):
        """Test ensemble factory function."""
        ensemble = create_ensemble()
        assert isinstance(ensemble, EnsembleModel)

    def test_ensemble_feature_importances(self, train_data):
        """Test ensemble feature importance."""
        X_train, y_train = train_data

        models = [
            ("xgboost", XGBoostModel(), 0.5),
            ("lightgbm", LightGBMModel(), 0.5),
        ]
        ensemble = EnsembleModel(models=models)
        ensemble.fit(X_train, y_train)

        importances = ensemble.feature_importances_
        assert len(importances) == X_train.shape[1]
