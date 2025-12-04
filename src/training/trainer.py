"""
Training Module

Provides comprehensive training pipeline with:
- Cross-validation
- Hyperparameter optimization (Optuna)
- Experiment tracking (MLflow)
- Early stopping
- Model checkpointing
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from src.config.settings import Settings, get_settings
from src.data.data_loader import DataLoader, create_data_loader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_pipeline import FeaturePipeline, create_feature_pipeline
from src.models.base_model import BaseModel, create_model
from src.models.ensemble import create_ensemble
from src.utils.helpers import ClassMapping, set_seed, timer
from src.utils.logger import get_logger
from src.utils.storage import ModelStorage

logger = get_logger(__name__)


class Trainer:
    """
    Main training orchestrator.

    Handles:
    - Data loading and preprocessing
    - Feature engineering
    - Model training
    - Cross-validation
    - Hyperparameter tuning
    - Experiment tracking
    """

    def __init__(
        self,
        settings: Settings | None = None,
        experiment_name: str | None = None,
        use_mlflow: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            settings: Application settings
            experiment_name: Name for experiment tracking
            use_mlflow: Whether to use MLflow
        """
        self.settings = settings or get_settings()
        self.experiment_name = (
            experiment_name or f"fqdn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.use_mlflow = use_mlflow and self.settings.mlflow.enabled

        # Set seed for reproducibility
        set_seed(self.settings.seed)

        # Initialize components
        self.data_loader: DataLoader | None = None
        self.preprocessor: DataPreprocessor | None = None
        self.feature_pipeline: FeaturePipeline | None = None
        self.model: BaseModel | None = None
        self.class_mapping: ClassMapping | None = None

        # Storage for artifacts
        self.storage = ModelStorage(self.settings.models_dir / "trained")

        # Metrics history
        self.metrics_history: list[dict[str, Any]] = []

        # MLflow run
        self._mlflow_run = None

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        if not self.use_mlflow:
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
            mlflow.set_experiment(self.settings.mlflow.experiment_name)

            self._mlflow_run = mlflow.start_run(run_name=self.experiment_name)

            # Log params
            mlflow.log_params(
                {
                    "model": self.settings.model.primary,
                    "seed": self.settings.seed,
                    "features_tfidf_max_features": self.settings.features.tfidf_max_features,
                }
            )

            # Enable autologging for supported libraries
            # This allows real-time metrics to be visible in MLflow UI
            try:
                if hasattr(mlflow, "xgboost"):
                    mlflow.xgboost.autolog(log_models=False, silent=True)
                if hasattr(mlflow, "lightgbm"):
                    mlflow.lightgbm.autolog(log_models=False, silent=True)
                if hasattr(mlflow, "catboost"):
                    mlflow.catboost.autolog(log_models=False, silent=True)
            except Exception as e:
                logger.warning(f"Failed to enable autologging: {e}")

            # Set tags
            for tag in self.settings.mlflow.tags:
                mlflow.set_tag("tag", tag)

            logger.info(f"MLflow initialized: {mlflow.get_tracking_uri()}")

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.use_mlflow = False

    def _log_mlflow(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        if self._mlflow_run:
            import mlflow

            mlflow.log_metrics(metrics, step=step)

    @timer("Loading data")
    def load_data(
        self,
        source: str = "auto",
        labeled_path: str | None = None,
        unlabeled_path: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load labeled and unlabeled data.

        Args:
            source: Data source type
            labeled_path: Path to labeled CSV
            unlabeled_path: Path to unlabeled CSV

        Returns:
            Tuple of (labeled_df, unlabeled_df)
        """
        self.data_loader = create_data_loader(source=source)

        # Override paths if provided
        if labeled_path and hasattr(self.data_loader._loader, "labeled_path"):
            self.data_loader._loader.labeled_path = Path(labeled_path)
        if unlabeled_path and hasattr(self.data_loader._loader, "unlabeled_path"):
            self.data_loader._loader.unlabeled_path = Path(unlabeled_path)

        labeled_df = self.data_loader.load_labeled()
        unlabeled_df = self.data_loader.load_unlabeled()

        logger.info(
            f"Loaded {len(labeled_df)} labeled, {len(unlabeled_df)} unlabeled records"
        )

        return labeled_df, unlabeled_df

    @timer("Preprocessing data")
    def preprocess(
        self,
        df: pl.DataFrame,
        fit: bool = True,
    ) -> pl.DataFrame:
        """
        Preprocess data.

        Args:
            df: DataFrame to preprocess
            fit: Whether to fit preprocessor

        Returns:
            Preprocessed DataFrame
        """
        if fit or self.preprocessor is None:
            self.preprocessor = DataPreprocessor(
                target_column=self.settings.target.column,
                orphan_value=self.settings.target.orphan_value,
                min_samples_per_class=self.settings.data.class_config.min_samples_per_class,
                handle_rare_classes=self.settings.data.class_config.handle_rare_classes,
                rare_class_threshold=self.settings.data.class_config.rare_class_threshold,
            )
            df = self.preprocessor.fit_transform(df)
        else:
            df = self.preprocessor.transform(df)

        self.class_mapping = self.preprocessor.class_mapping
        return df

    @timer("Engineering features")
    def engineer_features(
        self,
        df: pl.DataFrame,
        fit: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Engineer features from DataFrame.

        Args:
            df: Preprocessed DataFrame
            fit: Whether to fit feature pipeline

        Returns:
            Tuple of (X, y, feature_names)
        """
        if fit or self.feature_pipeline is None:
            self.feature_pipeline = create_feature_pipeline()

        # Get target
        target_col = self.settings.target.column
        y = df[target_col].to_numpy()

        # Transform features
        if fit:
            X, feature_names = self.feature_pipeline.fit_transform(df, y)
        else:
            X, feature_names = self.feature_pipeline.transform(df)

        # Encode labels
        if self.class_mapping:
            y_encoded = np.array(self.class_mapping.transform(y.tolist()))
        else:
            y_encoded = y

        logger.info(f"Feature matrix: {X.shape}, Target: {y_encoded.shape}")

        return X, y_encoded, feature_names

    @timer("Training model")
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        model_type: str = "ensemble",
    ) -> BaseModel:
        """
        Train model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model to train

        Returns:
            Trained model
        """
        self._init_mlflow()

        if model_type == "ensemble":
            self.model = create_ensemble()
        else:
            self.model = create_model(model_type)

        # Identify categorical features for CatBoost
        categorical_features = None
        if self.feature_pipeline and self.feature_pipeline.feature_names:
            # Get indices of categorical features
            # This assumes features from CategoricalEncoder are what we want
            # Note: CatBoost expects indices of columns in X that are categorical
            # Since our pipeline does one-hot or label encoding, we need to be careful.
            # If using label encoding (which is best for CatBoost), we should pass those indices.

            # For now, let's look at feature names that come from categorical encoder
            if (
                hasattr(self.feature_pipeline, "categorical_encoder")
                and self.feature_pipeline.categorical_encoder
            ):
                cat_feature_names = set(
                    self.feature_pipeline.categorical_encoder.feature_names
                )
                categorical_features = [
                    i
                    for i, name in enumerate(self.feature_pipeline.feature_names)
                    if name in cat_feature_names
                ]

        self.model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=self.feature_pipeline.feature_names
            if self.feature_pipeline
            else None,
            categorical_features=categorical_features,
        )

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
            self._log_mlflow({"val_" + k: v for k, v in val_metrics.items()})

        return self.model

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model on data."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        y_pred = self.model.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        }

        return metrics

    @timer("Cross-validation")
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        model_type: str = "ensemble",
    ) -> dict[str, list[float]]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            model_type: Model type

        Returns:
            Dictionary of metrics for each fold
        """
        self._init_mlflow()

        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.settings.seed
        )

        fold_metrics: dict[str, list[float]] = {
            "accuracy": [],
            "f1_macro": [],
            "f1_weighted": [],
            "precision_macro": [],
            "recall_macro": [],
        }

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and train model
            if model_type == "ensemble":
                model = create_ensemble()
            else:
                model = create_model(model_type)

            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)

            for key, value in metrics.items():
                fold_metrics[key].append(value)

            self._log_mlflow({f"cv_fold_{fold}_{k}": v for k, v in metrics.items()})

        # Log summary statistics
        summary = {}
        for key, values in fold_metrics.items():
            summary[f"cv/{key}_mean"] = np.mean(values)
            summary[f"cv/{key}_std"] = np.std(values)
            logger.info(f"CV {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        self._log_mlflow(summary)

        return fold_metrics

    def _evaluate_model(
        self, model: BaseModel, X: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """Evaluate a model."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        y_pred = model.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        }

    @timer("Hyperparameter optimization")
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        model_type: str = "xgboost",
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            model_type: Model type to optimize

        Returns:
            Best hyperparameters
        """
        import optuna
        from optuna.integration.mlflow import MLflowCallback

        def objective(trial: optuna.Trial) -> float:
            # Define search space based on model type
            if model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                }
            elif model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                }
            else:
                raise ValueError(f"Optimization not supported for {model_type}")

            # Create and train model
            model = create_model(model_type, **params)
            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            return metrics["f1_macro"]

        # Create study
        study = optuna.create_study(
            direction="maximize", study_name=self.experiment_name
        )

        # Add MLflow callback if enabled
        callbacks = []
        if self.use_mlflow:
            try:
                callbacks.append(
                    MLflowCallback(
                        tracking_uri=self.settings.mlflow.tracking_uri,
                        metric_name="f1_macro",
                    )
                )
            except Exception:
                pass

        # Optimize
        study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def save_model(
        self,
        model_name: str = "fqdn_classifier",
        version: str | None = None,
    ) -> str:
        """
        Save trained model and artifacts.

        Args:
            model_name: Name for the model
            version: Version string

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        artifacts = {}

        if self.preprocessor:
            artifacts["preprocessor"] = self.preprocessor

        if self.feature_pipeline:
            artifacts["feature_pipeline"] = self.feature_pipeline

        if self.class_mapping:
            artifacts["class_mapping"] = self.class_mapping

        metadata = {
            "experiment_name": self.experiment_name,
            "model_type": self.settings.model.primary,
            "n_features": (
                self.feature_pipeline.n_features if self.feature_pipeline else None
            ),
            "n_classes": self.model.n_classes if self.model._fitted else None,
            "settings": {
                "seed": self.settings.seed,
                "target_column": self.settings.target.column,
            },
        }

        path = self.storage.save(
            self.model,
            model_name,
            version=version,
            metadata=metadata,
            artifacts=artifacts,
        )

        # Log to MLflow
        if self._mlflow_run:
            import mlflow

            mlflow.log_artifacts(path)

        return path

    def finish(self) -> None:
        """Finish training and cleanup."""
        if self._mlflow_run:
            import mlflow

            mlflow.end_run()
            self._mlflow_run = None


def create_trainer(
    experiment_name: str | None = None,
    use_mlflow: bool = True,
) -> Trainer:
    """
    Factory function to create a trainer.

    Args:
        experiment_name: Experiment name
        use_mlflow: Whether to use MLflow

    Returns:
        Trainer instance
    """
    return Trainer(experiment_name=experiment_name, use_mlflow=use_mlflow)
