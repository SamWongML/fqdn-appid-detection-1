"""
Model Storage Utility

Provides utilities for saving and loading models, artifacts, and metadata.
Supports versioning and multiple serialization formats.
"""



import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelStorage:
    """
    Unified model storage and versioning system.

    Handles saving/loading of:
    - Trained models (joblib, pickle, ONNX)
    - Preprocessing pipelines
    - Feature transformers
    - Label encoders
    - Configuration metadata
    """

    def __init__(
        self,
        base_path: str | Path = "models/trained",
        versioning: Literal["semantic", "timestamp", "hash"] = "timestamp",
    ):
        """
        Initialize model storage.

        Args:
            base_path: Base directory for model storage
            versioning: Versioning strategy
        """
        self.base_path = Path(base_path)
        self.versioning = versioning
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Model registry file
        self.registry_path = self.base_path / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load model registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}, "latest": None}

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def _generate_version(
        self,
        model_name: str,
        model: Any = None,
    ) -> str:
        """
        Generate version string based on strategy.

        Args:
            model_name: Name of the model
            model: Model object (for hash versioning)

        Returns:
            Version string
        """
        if self.versioning == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")

        elif self.versioning == "semantic":
            # Get current version and increment
            current = self._registry["models"].get(model_name, {}).get("versions", [])
            if not current:
                return "1.0.0"

            latest = current[-1]["version"]
            parts = latest.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)

        elif self.versioning == "hash":
            # Hash model bytes
            data = joblib.dumps(model)
            return hashlib.sha256(data).hexdigest()[:12]

        raise ValueError(f"Unknown versioning strategy: {self.versioning}")

    def save(
        self,
        model: Any,
        model_name: str,
        version: str | None = None,
        metadata: dict[str, Any | None] = None,
        compress: int = 3,
        artifacts: dict[str, Any | None] = None,
    ) -> str:
        """
        Save a model with metadata and artifacts.

        Args:
            model: Model object to save
            model_name: Name identifier for the model
            version: Version string (auto-generated if None)
            metadata: Additional metadata to store
            compress: Compression level (0-9)
            artifacts: Additional artifacts to save (e.g., label encoder)

        Returns:
            Path to saved model directory
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name, model)

        # Create model directory
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save main model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path, compress=compress)
        logger.info(f"Saved model to {model_path}")

        # Save artifacts
        if artifacts:
            artifacts_dir = model_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            for name, artifact in artifacts.items():
                artifact_path = artifacts_dir / f"{name}.joblib"
                joblib.dump(artifact, artifact_path, compress=compress)
                logger.debug(f"Saved artifact '{name}' to {artifact_path}")

        # Prepare metadata
        full_metadata = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "model_module": type(model).__module__,
            "artifacts": list(artifacts.keys()) if artifacts else [],
            **(metadata or {}),
        }

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2, default=str)

        # Update registry
        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = {"versions": []}

        self._registry["models"][model_name]["versions"].append(
            {
                "version": version,
                "path": str(model_dir),
                "created_at": full_metadata["created_at"],
            }
        )
        self._registry["latest"] = str(model_dir)
        self._save_registry()

        logger.info(f"Model '{model_name}' v{version} saved to {model_dir}")
        return str(model_dir)

    def load(
        self,
        model_name: str,
        version: str | None = None,
        load_artifacts: bool = True,
    ) -> dict[str, Any]:
        """
        Load a model with its artifacts and metadata.

        Args:
            model_name: Name of the model to load
            version: Specific version (None for latest)
            load_artifacts: Whether to load artifacts

        Returns:
            Dictionary with model, artifacts, and metadata
        """
        # Get model directory
        if version is None:
            # Get latest version
            versions = self._registry["models"].get(model_name, {}).get("versions", [])
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            version = versions[-1]["version"]

        model_dir = self.base_path / model_name / version

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Load model
        model_path = model_dir / "model.joblib"
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load artifacts
        artifacts = {}
        if load_artifacts:
            artifacts_dir = model_dir / "artifacts"
            if artifacts_dir.exists():
                for artifact_file in artifacts_dir.glob("*.joblib"):
                    name = artifact_file.stem
                    artifacts[name] = joblib.load(artifact_file)
                    logger.debug(f"Loaded artifact '{name}'")

        return {
            "model": model,
            "artifacts": artifacts,
            "metadata": metadata,
            "version": version,
            "path": str(model_dir),
        }

    def list_models(self) -> dict[str, list[str]]:
        """
        List all available models and versions.

        Returns:
            Dictionary mapping model names to version lists
        """
        return {
            name: [v["version"] for v in info["versions"]]
            for name, info in self._registry["models"].items()
        }

    def get_latest_version(self, model_name: str) -> str | None:
        """
        Get the latest version of a model.

        Args:
            model_name: Name of the model

        Returns:
            Latest version string or None
        """
        versions = self._registry["models"].get(model_name, {}).get("versions", [])
        return versions[-1]["version"] if versions else None

    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            True if deleted, False if not found
        """
        model_dir = self.base_path / model_name / version

        if not model_dir.exists():
            return False

        # Remove from disk
        shutil.rmtree(model_dir)

        # Update registry
        if model_name in self._registry["models"]:
            self._registry["models"][model_name]["versions"] = [
                v
                for v in self._registry["models"][model_name]["versions"]
                if v["version"] != version
            ]
            self._save_registry()

        logger.info(f"Deleted model '{model_name}' v{version}")
        return True

    def export_onnx(
        self,
        model_name: str,
        version: str | None = None,
        sample_input: Any | None = None,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            model_name: Name of the model
            version: Version to export
            sample_input: Sample input for tracing

        Returns:
            Path to ONNX file
        """
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("skl2onnx required for ONNX export")

        loaded = self.load(model_name, version, load_artifacts=False)
        model = loaded["model"]
        model_dir = Path(loaded["path"])

        # Determine input shape
        if sample_input is not None:
            n_features = sample_input.shape[1]
        else:
            # Try to infer from model
            n_features = getattr(model, "n_features_in_", None)
            if n_features is None:
                raise ValueError("Cannot determine input shape. Provide sample_input.")

        initial_type = [("input", FloatTensorType([None, n_features]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Save
        onnx_path = model_dir / "model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Exported ONNX model to {onnx_path}")
        return str(onnx_path)


# ============================================================================
# Convenience Functions
# ============================================================================

_default_storage: ModelStorage | None = None


def get_storage(base_path: str | Path = "models/trained") -> ModelStorage:
    """Get or create default storage instance."""
    global _default_storage
    if _default_storage is None:
        _default_storage = ModelStorage(base_path)
    return _default_storage


def save_model(
    model: Any,
    model_name: str,
    version: str | None = None,
    metadata: dict[str, Any | None] = None,
    artifacts: dict[str, Any | None] = None,
    base_path: str | Path = "models/trained",
) -> str:
    """
    Save a model (convenience function).

    Args:
        model: Model to save
        model_name: Model identifier
        version: Version string
        metadata: Additional metadata
        artifacts: Additional artifacts
        base_path: Storage base path

    Returns:
        Path to saved model
    """
    storage = ModelStorage(base_path)
    return storage.save(model, model_name, version, metadata, artifacts=artifacts)


def load_model(
    model_name: str,
    version: str | None = None,
    base_path: str | Path = "models/trained",
) -> dict[str, Any]:
    """
    Load a model (convenience function).

    Args:
        model_name: Model identifier
        version: Specific version (None for latest)
        base_path: Storage base path

    Returns:
        Dictionary with model, artifacts, metadata
    """
    storage = ModelStorage(base_path)
    return storage.load(model_name, version)


def save_artifact(
    artifact: Any,
    name: str,
    path: str | Path,
    compress: int = 3,
) -> str:
    """
    Save a single artifact.

    Args:
        artifact: Object to save
        name: Artifact name
        path: Directory to save to
        compress: Compression level

    Returns:
        Path to saved artifact
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    artifact_path = path / f"{name}.joblib"
    joblib.dump(artifact, artifact_path, compress=compress)
    logger.debug(f"Saved artifact '{name}' to {artifact_path}")

    return str(artifact_path)


def load_artifact(
    name: str,
    path: str | Path,
) -> Any:
    """
    Load a single artifact.

    Args:
        name: Artifact name
        path: Directory containing artifact

    Returns:
        Loaded artifact
    """
    artifact_path = Path(path) / f"{name}.joblib"
    artifact = joblib.load(artifact_path)
    logger.debug(f"Loaded artifact '{name}' from {artifact_path}")
    return artifact
