"""
ScalForex — Structured Logger
Centralized logging for the entire project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "scalforex",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Create a structured logger with console + optional file output.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# Default project logger
log = setup_logger()
