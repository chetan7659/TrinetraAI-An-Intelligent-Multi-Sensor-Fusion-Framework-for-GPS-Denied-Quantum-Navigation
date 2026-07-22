"""Centralized logging system for Trinetra-AI."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from trinetra_core.utils.config import get_config

# Resolve the project root assuming src/trinetra_core/utils/logger.py -> ../../../../
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def setup_logger(name: str) -> logging.Logger:
    """Create and configure a logger for a specific module.

    Reads configuration from configs/logging.yaml to determine:
    - Log level
    - Whether to log to the console
    - Whether to save logs to a file and the directory path

    Args:
        name: The name of the module requesting the logger (usually __name__).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, return it to prevent duplicate logs
    if logger.handlers:
        return logger

    # Load configuration
    try:
        config = get_config("logging")
    except FileNotFoundError:
        # Fallback defaults if config is missing
        config = {
            "log_level": "INFO",
            "console_logging": True,
            "save_logs": False,
            "log_directory": "logs/",
        }

    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Define standard format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Handler
    if config.get("console_logging", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 2. File Handler
    if config.get("save_logs", True):
        log_dir_str = config.get("log_directory", "logs/")
        # Ensure log_dir is absolute based on project root
        log_dir = _PROJECT_ROOT / log_dir_str
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"trinetra_{date_str}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log propagation to the root logger to avoid double-logging
    logger.propagate = False

    return logger
