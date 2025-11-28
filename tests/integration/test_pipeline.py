"""Integration tests for end-to-end pipeline."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.evaluation.evaluator import Evaluator
from src.features.feature_pipeline import FeaturePipeline
from src.inference.predictor import Predictor
from src.models.base_model import XGBoostModel
from src.models.ensemble import EnsembleModel, create_ensemble


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    @pytest.fixture
    def labeled_data(self):
        """Generate labeled training data."""
        np.random.seed(42)
        n_samples = 500

        # Generate FQDNs with patterns
        fqdns = []
        appids = []
        record_types = []
        brands = []

        templates = [
            ("api.{env}.{brand}.com", ["dev", "staging", "prod"]),
            ("www.{brand}.{region}.co.uk", ["uk", "us", "eu"]),
            ("cdn.{service}.{brand}.net", ["static", "media", "assets"]),
            ("mail.{brand}.internal.org", ["corporate", "personal"]),
            ("db.{dbtype}.{brand}.cloud", ["mysql", "postgres", "mongo"]),
        ]

        brands_list = ["acme", "globex", "initech", "umbrella", "wayne"]

        for i in range(n_samples):
            template, options = templates[i % len(templates)]
            brand = brands_list[i % len(brands_list)]
            option = options[i % len(options)]

            fqdn = template.format(
                env=option if "env" in template else "",
                brand=brand,
                region=option if "region" in template else "",
                service=option if "service" in template else "",
                dbtype=option if "dbtype" in template else "",
            ).replace("..", ".")

            fqdns.append(fqdn)
            appids.append((i % 10) + 1000)  # 10 different appIds
            record_types.append(["A", "CNAME", "MX"][i % 3])
            brands.append(brand)

        return pl.DataFrame(
            {
                "fqdn": fqdns,
                "appid": appids,
                "record_type": record_types,
                "brand": brands,
                "record_data": [f"192.168.{i%256}.{i%256}" for i in range(n_samples)],
            }
        )

    @pytest.fixture
    def orphan_data(self):
        """Generate orphan (unlabeled) data."""
        return pl.DataFrame(
            {
                "fqdn": [
                    "unknown.mystery.service.com",
                    "new.api.globex.staging.net",
                    "cdn.static.newbrand.io",
                ],
                "appid": [0, 0, 0],
                "record_type": ["A", "CNAME", "A"],
                "record_data": ["10.0.0.1", "proxy.net", "10.0.0.2"],
                "brand": ["unknown", "globex", "newbrand"],
            }
        )

    def test_full_training_pipeline(self, labeled_data, tmp_path):
        """Test complete training pipeline."""
        # 1. Preprocess
        preprocessor = DataPreprocessor(
            target_column="appid",
            min_samples_per_class=5,
        )
        processed_df = preprocessor.fit_transform(labeled_data)

        assert len(processed_df) > 0

        # 2. Feature engineering
        feature_pipeline = FeaturePipeline(
            fqdn_column="fqdn",
            target_column="appid",
            categorical_columns=["record_type", "brand"],
            tfidf_max_features=100,
        )

        y = processed_df["appid"].to_numpy()
        if preprocessor.class_mapping:
            y = np.array(preprocessor.class_mapping.transform(y.tolist()))

        X, feature_names = feature_pipeline.fit_transform(processed_df, y)

        assert X.shape[0] == len(processed_df)
        assert len(feature_names) > 0

        # 3. Train/test split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 4. Train model
        model = XGBoostModel()
        model.fit(X_train, y_train, X_test, y_test)

        assert model.is_fitted

        # 5. Evaluate
        evaluator = Evaluator(model, feature_names)
        results = evaluator.evaluate(X_test, y_test)

        assert results.accuracy > 0
        assert results.f1_macro > 0

        # Should perform reasonably well on synthetic data
        assert results.accuracy > 0.35

    def test_prediction_pipeline(self, labeled_data, orphan_data, tmp_path):
        """Test complete prediction pipeline."""
        # Train pipeline
        preprocessor = DataPreprocessor(target_column="appid", min_samples_per_class=5)
        processed_df = preprocessor.fit_transform(labeled_data)

        feature_pipeline = FeaturePipeline(
            fqdn_column="fqdn",
            target_column="appid",
            tfidf_max_features=100,
        )

        y = processed_df["appid"].to_numpy()
        y_encoded = np.array(preprocessor.class_mapping.transform(y.tolist()))

        X, _ = feature_pipeline.fit_transform(processed_df, y_encoded)

        model = XGBoostModel()
        model.fit(X, y_encoded)

        # Create predictor
        predictor = Predictor(
            model=model,
            preprocessor=preprocessor,
            feature_pipeline=feature_pipeline,
            class_mapping=preprocessor.class_mapping,
            confidence_threshold=0.3,
            top_k=3,
        )

        # Predict on orphan data
        results = predictor.predict_batch(orphan_data)

        assert results.total_count == len(orphan_data)
        assert len(results.predictions) == len(orphan_data)

        # Check prediction format
        for pred in results.predictions:
            assert pred.fqdn is not None
            assert pred.predicted_appid is not None
            assert 0 <= pred.confidence <= 1
            assert len(pred.top_k_predictions) <= 3

    def test_ensemble_pipeline(self, labeled_data):
        """Test pipeline with ensemble model."""
        # Preprocess
        preprocessor = DataPreprocessor(target_column="appid", min_samples_per_class=5)
        processed_df = preprocessor.fit_transform(labeled_data)

        # Feature engineering
        feature_pipeline = FeaturePipeline(
            fqdn_column="fqdn",
            tfidf_max_features=100,
        )

        y = processed_df["appid"].to_numpy()
        y_encoded = np.array(preprocessor.class_mapping.transform(y.tolist()))

        X, _ = feature_pipeline.fit_transform(processed_df, y_encoded)

        # Split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        # Train ensemble with just 2 models for speed
        from src.models.base_model import LightGBMModel

        models = [
            ("xgboost", XGBoostModel(), 0.5),
            ("lightgbm", LightGBMModel(), 0.5),
        ]
        ensemble = EnsembleModel(models=models)
        ensemble.fit(X_train, y_train, X_test, y_test)

        # Evaluate
        evaluator = Evaluator(ensemble)
        results = evaluator.evaluate(X_test, y_test)

        assert results.accuracy > 0.35

    def test_cross_validation(self, labeled_data):
        """Test cross-validation integration."""
        from sklearn.model_selection import StratifiedKFold

        # Prepare data
        preprocessor = DataPreprocessor(target_column="appid", min_samples_per_class=5)
        processed_df = preprocessor.fit_transform(labeled_data)

        feature_pipeline = FeaturePipeline(
            fqdn_column="fqdn",
            tfidf_max_features=100,
        )

        y = processed_df["appid"].to_numpy()
        y_encoded = np.array(preprocessor.class_mapping.transform(y.tolist()))

        X, _ = feature_pipeline.fit_transform(processed_df, y_encoded)

        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            model = XGBoostModel()
            model.fit(X_train, y_train, X_val, y_val)

            y_pred = model.predict(X_val)
            accuracy = (y_pred == y_val).mean()
            fold_accuracies.append(accuracy)

        # All folds should have reasonable accuracy
        assert all(acc > 0.3 for acc in fold_accuracies)

        # Average should be decent
        avg_accuracy = np.mean(fold_accuracies)
        assert avg_accuracy > 0.4


class TestModelPersistence:
    """Tests for model persistence and loading."""

    def test_save_and_load_artifacts(self, tmp_path):
        """Test saving and loading all artifacts."""
        from src.utils.storage import ModelStorage

        # Create mock components
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 5, 100)

        model = XGBoostModel()
        model.fit(X, y)

        # Save
        storage = ModelStorage(tmp_path)
        path = storage.save(
            model,
            "test_model",
            metadata={"version": "1.0"},
        )

        # Load
        loaded = storage.load("test_model")

        assert loaded["model"].is_fitted
        assert loaded["metadata"]["version"] == "1.0"

        # Verify predictions match
        original_pred = model.predict(X[:10])
        loaded_pred = loaded["model"].predict(X[:10])

        np.testing.assert_array_equal(original_pred, loaded_pred)
