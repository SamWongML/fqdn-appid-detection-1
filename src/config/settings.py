"""
Configuration Settings Module

Provides centralized configuration management using Pydantic for validation
and YAML files for persistence. Supports environment variable overrides.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()

# ============================================================================
# Helper Functions
# ============================================================================


def resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively resolve environment variable references in config values.

    Supports format: ${VAR_NAME:default_value}

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with resolved environment variables
    """
    env_pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            match = env_pattern.match(value)
            if match:
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.environ.get(var_name, default)
            return value
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value

    return resolve_value(config)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve environment variables
    config = resolve_env_vars(config)

    return config


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    result = {}

    for config in configs:
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result


# ============================================================================
# Settings Models
# ============================================================================


class DatabaseSettings(BaseModel):
    """PostgreSQL database connection settings."""

    host: str = "localhost"
    port: int = 5432
    database: str = "fqdn_db"
    schema_name: str = Field(default="da", alias="schema")
    table: str = "fqdn_prod_inventory_enriched_v2"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    connection_timeout: int = 30
    chunk_size: int = 50000
    use_arrow: bool = True

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def connectorx_uri(self) -> str:
        """Generate ConnectorX connection URI for fast transfers."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class CSVSettings(BaseModel):
    """CSV data source settings."""

    labeled_data: str = "data/raw/labeled_fqdns.csv"
    unlabeled_data: str = "data/raw/orphan_fqdns.csv"
    separator: str = ","
    has_header: bool = True
    encoding: str = "utf-8"
    null_values: list[str] = ["", "NULL", "null", "None", "NA", "N/A"]
    ignore_errors: bool = False


class SplitSettings(BaseModel):
    """Train/validation/test split settings."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_column: str = "appid"
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_ratios(self) -> "SplitSettings":
        """Validate that split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class ClassConfigSettings(BaseModel):
    """Class configuration for handling imbalanced data."""

    min_samples_per_class: int = 3
    max_classes: int | None = None
    handle_rare_classes: Literal["drop", "group", "oversample"] = "group"
    rare_class_threshold: int = 5


class DataSettings(BaseModel):
    """Data loading and processing settings."""

    primary_source: Literal["postgres", "csv"] = "csv"
    postgres: DatabaseSettings = Field(default_factory=DatabaseSettings)
    csv: CSVSettings = Field(default_factory=CSVSettings)
    split: SplitSettings = Field(default_factory=SplitSettings)
    class_config: ClassConfigSettings = Field(default_factory=ClassConfigSettings)

    # Filtering
    include_status: list[str] = ["Active", "Demised"]
    exclude_patterns: list[str] = []


class TargetSettings(BaseModel):
    """Target variable settings."""

    column: str = "appid"
    orphan_value: int = 0
    unknown_class_label: int = -1


class XGBoostSettings(BaseModel):
    """XGBoost model hyperparameters."""

    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    tree_method: str = "hist"
    device: str = "cpu"
    n_estimators: int = 500
    max_depth: int = 8
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 0.8
    learning_rate: float = 0.05
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    max_delta_step: int = 1
    scale_pos_weight: float = 1.0
    verbosity: int = 1


class LightGBMSettings(BaseModel):
    """LightGBM model hyperparameters."""

    objective: str = "multiclass"
    metric: str = "multi_logloss"
    boosting_type: str = "gbdt"
    device: str = "cpu"
    n_estimators: int = 500
    num_leaves: int = 128
    max_depth: int = -1
    min_child_samples: int = 20
    min_child_weight: float = 0.001
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    learning_rate: float = 0.05
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    categorical_feature: str = "auto"
    num_threads: int = -1
    verbose: int = -1
    class_weight: str | dict | None = None
    is_unbalance: bool = False


class CatBoostSettings(BaseModel):
    """CatBoost model hyperparameters."""

    loss_function: str = "MultiClass"
    eval_metric: str = "MultiClass"
    task_type: str = "CPU"
    iterations: int = 500
    depth: int = 8
    min_data_in_leaf: int = 10
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    one_hot_max_size: int = 25
    early_stopping_rounds: int = 50
    verbose: int = 100


class EnsembleModelConfig(BaseModel):
    """Single model configuration in ensemble."""

    name: str
    weight: float = 1.0


class EnsembleSettings(BaseModel):
    """Ensemble model settings."""

    models: list[EnsembleModelConfig] = [
        EnsembleModelConfig(name="xgboost", weight=0.4),
        EnsembleModelConfig(name="lightgbm", weight=0.4),
        EnsembleModelConfig(name="catboost", weight=0.2),
    ]
    voting: Literal["hard", "soft"] = "soft"


class ModelSettings(BaseModel):
    """Model configuration settings."""

    primary: Literal["xgboost", "lightgbm", "catboost", "ensemble", "neural"] = (
        "ensemble"
    )
    ensemble: EnsembleSettings = Field(default_factory=EnsembleSettings)
    xgboost: XGBoostSettings = Field(default_factory=XGBoostSettings)
    lightgbm: LightGBMSettings = Field(default_factory=LightGBMSettings)
    catboost: CatBoostSettings = Field(default_factory=CatBoostSettings)


class OptimizationSettings(BaseModel):
    """Hyperparameter optimization settings."""

    enabled: bool = True
    n_trials: int = 100
    timeout: int = 3600
    objective_metric: str = "f1_macro"
    direction: Literal["minimize", "maximize"] = "maximize"
    pruner: str = "median"
    n_warmup_steps: int = 10


class CrossValidationSettings(BaseModel):
    """Cross-validation settings."""

    strategy: Literal["stratified_kfold", "kfold", "time_series"] = "stratified_kfold"
    n_splits: int = 5
    shuffle: bool = True


class OpenSetSettings(BaseModel):
    """Open-set recognition settings."""

    enabled: bool = True
    confidence_threshold: float = 0.3
    entropy_threshold: float = 2.0
    fallback_enabled: bool = True
    fallback_method: Literal["embedding_similarity", "rule_based"] = (
        "embedding_similarity"
    )
    similarity_threshold: float = 0.85


class InferenceSettings(BaseModel):
    """Inference pipeline settings."""

    batch_size: int = 1000
    include_probabilities: bool = True
    include_top_k: bool = True
    top_k: int = 5
    include_confidence: bool = True
    include_explanation: bool = False
    cache_predictions: bool = True
    cache_ttl: int = 3600


class MLflowSettings(BaseModel):
    """MLflow experiment tracking settings."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "fqdn-orphan-detection"
    tags: list[str] = ["fqdn", "classification", "production"]
    log_artifacts: list[str] = ["model", "config", "metrics", "confusion_matrix"]


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    file_enabled: bool = True
    file_path: str = "logs/fqdn_detection.log"
    file_rotation: str = "10 MB"
    file_retention: str = "30 days"
    console_enabled: bool = True
    console_colorize: bool = True


class FeatureGroupSettings(BaseModel):
    """Settings for a feature group."""

    enabled: bool = True
    features: list[str] = []
    priority_weight: float = 1.0


class FeatureSettings(BaseModel):
    """Feature engineering settings."""

    # Feature groups
    primary: FeatureGroupSettings = Field(
        default_factory=lambda: FeatureGroupSettings(
            features=[
                "fqdn",
                "record_type",
                "record_data",
                "fqdn_source",
                "fqdn_status",
            ]
        )
    )
    domain_derived: FeatureGroupSettings = Field(
        default_factory=lambda: FeatureGroupSettings(
            features=[
                "bus_domain",
                "bus_sub_domain",
                "domain_name",
                "country_code",
                "num_dots",
            ]
        )
    )
    business: FeatureGroupSettings = Field(
        default_factory=lambda: FeatureGroupSettings(
            features=["brand", "product", "market", "market_group", "client_channel"]
        )
    )
    override: FeatureGroupSettings = Field(
        default_factory=lambda: FeatureGroupSettings(
            features=[
                "override_appid",
                "override_brand",
                "override_product",
                "override_market",
            ],
            priority_weight=2.0,
        )
    )

    # Text feature settings
    tfidf_max_features: int = 1000
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    ngram_min: int = 2
    ngram_max: int = 4
    char_ngram_max_features: int = 500


# ============================================================================
# Main Settings Class
# ============================================================================


class Settings(BaseSettings):
    """
    Main application settings.

    Loads configuration from environment variables and YAML files.
    Environment variables take precedence.
    """

    model_config = SettingsConfigDict(
        env_prefix="FQDN_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Project info
    project_name: str = "fqdn-orphan-detection"
    version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    # Sub-settings (loaded from YAML)
    data: DataSettings = Field(default_factory=DataSettings)
    target: TargetSettings = Field(default_factory=TargetSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    cross_validation: CrossValidationSettings = Field(
        default_factory=CrossValidationSettings
    )
    open_set: OpenSetSettings = Field(default_factory=OpenSetSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        feature_config_path: str | Path | None = None,
    ) -> "Settings":
        """
        Create settings from YAML configuration files.

        Args:
            config_path: Path to main config.yaml
            model_config_path: Path to model_config.yaml
            feature_config_path: Path to feature_config.yaml

        Returns:
            Settings instance
        """
        config = {}

        # Load main config
        if config_path and Path(config_path).exists():
            main_config = load_yaml_config(config_path)
            config = merge_configs(config, main_config)

        # Load model config
        if model_config_path and Path(model_config_path).exists():
            model_cfg = load_yaml_config(model_config_path)
            config = merge_configs(config, {"model": model_cfg.get("model", {})})
            if "xgboost" in model_cfg:
                config = merge_configs(
                    config, {"model": {"xgboost": model_cfg["xgboost"]}}
                )
            if "lightgbm" in model_cfg:
                config = merge_configs(
                    config, {"model": {"lightgbm": model_cfg["lightgbm"]}}
                )
            if "catboost" in model_cfg:
                config = merge_configs(
                    config, {"model": {"catboost": model_cfg["catboost"]}}
                )
            if "optimization" in model_cfg:
                config = merge_configs(
                    config, {"optimization": model_cfg["optimization"]}
                )
            if "cross_validation" in model_cfg:
                config = merge_configs(
                    config, {"cross_validation": model_cfg["cross_validation"]}
                )
            if "open_set" in model_cfg:
                config = merge_configs(config, {"open_set": model_cfg["open_set"]})
            if "inference" in model_cfg:
                config = merge_configs(config, {"inference": model_cfg["inference"]})

        # Load feature config
        if feature_config_path and Path(feature_config_path).exists():
            feature_cfg = load_yaml_config(feature_config_path)
            if "feature_groups" in feature_cfg:
                config = merge_configs(
                    config, {"features": feature_cfg["feature_groups"]}
                )

        return cls(**config)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "external").mkdir(exist_ok=True)
        (self.models_dir / "trained").mkdir(exist_ok=True)
        (self.models_dir / "experiments").mkdir(exist_ok=True)


# ============================================================================
# Settings Singleton
# ============================================================================


@lru_cache()
def get_settings(
    config_path: str | None = None,
    model_config_path: str | None = None,
    feature_config_path: str | None = None,
) -> Settings:
    """
    Get application settings (cached singleton).

    Args:
        config_path: Path to main config.yaml
        model_config_path: Path to model_config.yaml
        feature_config_path: Path to feature_config.yaml

    Returns:
        Settings instance
    """
    # Default paths
    config_dir = Path("config")

    if config_path is None and (config_dir / "config.yaml").exists():
        config_path = str(config_dir / "config.yaml")

    if model_config_path is None and (config_dir / "model_config.yaml").exists():
        model_config_path = str(config_dir / "model_config.yaml")

    if feature_config_path is None and (config_dir / "feature_config.yaml").exists():
        feature_config_path = str(config_dir / "feature_config.yaml")

    return Settings.from_yaml(
        config_path=config_path,
        model_config_path=model_config_path,
        feature_config_path=feature_config_path,
    )


def reset_settings_cache() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
