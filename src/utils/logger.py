"""
Logging Utility Module

Provides structured logging using loguru with support for:
- Console output with colors
- File rotation
- JSON formatting for production
- Context injection
"""



import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


def json_formatter(record: dict[str, Any]) -> str:
    """
    Format log record as JSON.

    Args:
        record: Loguru record dictionary

    Returns:
        JSON formatted string
    """
    import json
    from datetime import datetime

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add extra context if present
    if record.get("extra"):
        log_entry["extra"] = record["extra"]

    # Add exception info if present
    if record.get("exception"):
        log_entry["exception"] = {
            "type": (
                record["exception"].type.__name__ if record["exception"].type else None
            ),
            "value": (
                str(record["exception"].value) if record["exception"].value else None
            ),
            "traceback": (
                record["exception"].traceback if record["exception"].traceback else None
            ),
        }

    return json.dumps(log_entry) + "\n"


def setup_logger(
    level: str = "INFO",
    log_format: str = "text",
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    colorize: bool = True,
    serialize: bool = False,
) -> None:
    """
    Configure the global logger.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("text" or "json")
        log_file: Path to log file (None for console only)
        rotation: When to rotate log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files
        colorize: Whether to colorize console output
        serialize: Whether to serialize logs to JSON
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if log_format == "json":
        logger.add(
            sys.stderr,
            format=json_formatter,
            level=level,
            colorize=False,
            serialize=serialize,
        )
    else:
        # Rich console handler
        logger.add(
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                markup=True,
                show_path=False,  # Loguru handles this
                enable_link_path=True,
            ),
            format="{message}",
            level=level,
            colorize=colorize,
        )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_format == "json":
            logger.add(
                str(log_path),
                format=json_formatter,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
                serialize=serialize,
            )
        else:
            logger.add(
                str(log_path),
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | "
                    "{module}:{function}:{line} - {message}"
                ),
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
            )


@lru_cache()
def get_logger(name: str | None = None) -> "logger":
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerContextManager:
    """Context manager for adding temporary context to logs."""

    def __init__(self, **context: Any):
        """
        Initialize with context to add.

        Args:
            **context: Key-value pairs to add to log context
        """
        self.context = context
        self._token = None

    def __enter__(self) -> "logger":
        """Enter context and bind context variables."""
        self._token = logger.contextualize(**self.context)
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and remove bound variables."""
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)


def log_context(**context: Any) -> LoggerContextManager:
    """
    Create a context manager for adding context to logs.

    Usage:
        with log_context(request_id="123", user="john"):
            logger.info("Processing request")

    Args:
        **context: Key-value pairs to add to log context

    Returns:
        LoggerContextManager instance
    """
    return LoggerContextManager(**context)


class ProgressLogger:
    """Logger wrapper for progress tracking."""

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        log_interval: int = 10,
    ):
        """
        Initialize progress logger.

        Args:
            total: Total number of items
            description: Progress description
            log_interval: How often to log (percentage)
        """
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.current = 0
        self._last_logged = 0

    def update(self, n: int = 1) -> None:
        """
        Update progress.

        Args:
            n: Number of items processed
        """
        self.current += n
        percentage = int(100 * self.current / self.total) if self.total > 0 else 100

        # Log at intervals
        if (
            percentage >= self._last_logged + self.log_interval
            or self.current >= self.total
        ):
            logger.info(
                f"{self.description}: {self.current}/{self.total} ({percentage}%)"
            )
            self._last_logged = percentage

    def __enter__(self) -> "ProgressLogger":
        """Enter context."""
        logger.info(f"{self.description}: Starting (0/{self.total})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        if exc_type is None:
            logger.info(f"{self.description}: Completed ({self.current}/{self.total})")
        else:
            logger.warning(f"{self.description}: Failed at {self.current}/{self.total}")


# Convenience exports
info = logger.info
debug = logger.debug
warning = logger.warning
error = logger.error
critical = logger.critical
exception = logger.exception
