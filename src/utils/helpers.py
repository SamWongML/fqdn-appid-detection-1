"""
Helper Utility Functions

Common utility functions used throughout the FQDN Orphan Detection system.
"""



import hashlib
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

import numpy as np
import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug(f"Random seed set to {seed}")


@contextmanager
def timer(description: str = "Operation") -> Generator[None, None, None]:
    """
    Context manager for timing code blocks.

    Args:
        description: Description of the operation being timed

    Yields:
        None

    Usage:
        with timer("Data loading"):
            load_data()
    """
    start_time = time.perf_counter()
    logger.info(f"{description}: Starting...")

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time

        if elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.2f}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            time_str = f"{hours}h {minutes}m {seconds:.2f}s"

        logger.info(f"{description}: Completed in {time_str}")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with timer(f"{func.__module__}.{func.__name__}"):
            return func(*args, **kwargs)

    return wrapper


def memory_usage() -> dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory usage in MB
    """
    import psutil

    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        "rss_mb": mem_info.rss / (1024**2),
        "vms_mb": mem_info.vms / (1024**2),
        "percent": process.memory_percent(),
    }


def log_memory(description: str = "Memory usage") -> None:
    """
    Log current memory usage.

    Args:
        description: Description prefix
    """
    try:
        usage = memory_usage()
        logger.info(
            f"{description}: RSS={usage['rss_mb']:.1f}MB, "
            f"VMS={usage['vms_mb']:.1f}MB, "
            f"Percent={usage['percent']:.1f}%"
        )
    except ImportError:
        logger.warning("psutil not installed, cannot log memory usage")


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_dataframe(df: pl.DataFrame, columns: list[str | None] = None) -> str:
    """
    Compute a hash of a Polars DataFrame for caching/versioning.

    Args:
        df: DataFrame to hash
        columns: Specific columns to include (None for all)

    Returns:
        SHA256 hash string
    """
    if columns:
        df = df.select(columns)

    # Convert to bytes and hash
    data = df.write_csv().encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def safe_divide(
    numerator: int | float | np.ndarray,
    denominator: int | float | np.ndarray,
    default: float = 0.0,
) -> float | np.ndarray:
    """
    Safely divide, returning default value for division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return when denominator is zero

    Returns:
        Division result or default
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(denominator != 0, numerator / denominator, default)
        return result
    elif denominator == 0:
        return default
    return numerator / denominator


def chunk_list(lst: list[T], chunk_size: int) -> Generator[list[T], None, None]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Yields:
        List chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys

    Returns:
        Flattened dictionary
    """
    items: list[tuple[str, Any]] = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """
    Unflatten a flattened dictionary.

    Args:
        d: Flattened dictionary
        sep: Separator used in keys

    Returns:
        Nested dictionary
    """
    result: dict[str, Any] = {}

    for key, value in d.items():
        parts = key.split(sep)
        target = result

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    return result


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.

    Args:
        fmt: datetime format string

    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(fmt)


def get_file_hash(filepath: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hash string
    """
    hasher = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying a function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff: Multiplier for delay after each attempt
        exceptions: Exceptions to catch and retry on

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception  # type: ignore

        return wrapper

    return decorator


def format_bytes(size: int) -> str:
    """
    Format byte size as human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_number(n: int | float, precision: int = 2) -> str:
    """
    Format number with thousands separator.

    Args:
        n: Number to format
        precision: Decimal precision for floats

    Returns:
        Formatted string
    """
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.{precision}f}"


class ClassMapping:
    """Bidirectional mapping between class labels and indices."""

    def __init__(self, classes: list[Any | None] = None):
        """
        Initialize class mapping.

        Args:
            classes: Optional list of class labels
        """
        self._label_to_idx: dict[Any, int] = {}
        self._idx_to_label: dict[int, Any] = {}

        if classes:
            for idx, label in enumerate(classes):
                self._label_to_idx[label] = idx
                self._idx_to_label[idx] = label

    def fit(self, labels: list[Any]) -> "ClassMapping":
        """
        Fit mapping from unique labels.

        Args:
            labels: List of labels

        Returns:
            self
        """
        unique_labels = sorted(set(labels))
        self._label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self._idx_to_label = {idx: label for label, idx in self._label_to_idx.items()}
        return self

    def label_to_idx(self, label: Any) -> int:
        """Get index for label."""
        return self._label_to_idx[label]

    def idx_to_label(self, idx: int) -> Any:
        """Get label for index."""
        return self._idx_to_label[idx]

    def transform(self, labels: list[Any]) -> list[int]:
        """Transform labels to indices."""
        return [self._label_to_idx[l] for l in labels]

    def inverse_transform(self, indices: list[int]) -> list[Any]:
        """Transform indices to labels."""
        return [self._idx_to_label[i] for i in indices]

    @property
    def classes(self) -> list[Any]:
        """Get ordered list of classes."""
        return [self._idx_to_label[i] for i in range(len(self._idx_to_label))]

    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        return len(self._label_to_idx)

    def save(self, path: str | Path) -> None:
        """Save mapping to file."""
        import json

        with open(path, "w") as f:
            json.dump(
                {
                    "label_to_idx": {str(k): v for k, v in self._label_to_idx.items()},
                    "idx_to_label": {str(k): v for k, v in self._idx_to_label.items()},
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ClassMapping":
        """Load mapping from file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        mapping = cls()
        mapping._label_to_idx = {int(k): v for k, v in data["label_to_idx"].items()}
        mapping._idx_to_label = {int(k): v for k, v in data["idx_to_label"].items()}
        return mapping
