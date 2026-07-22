"""Centralized logging system for Trinetra-AI."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from trinetra.infrastructure.config import get_config

# Resolve the project root: src/trinetra/infrastructure/logging/logger.py -> ../../../../../
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def setup_logger(name: str) -> logging.Logger:
    """Create and configure a logger for a specific module.

    Reads configuration from configs/logging.yaml.

    Args:
        name: The name of the module requesting the logger.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    try:
        config = get_config("logging")
    except FileNotFoundError:
        config = {
            "log_level": "INFO",
            "console_logging": True,
            "save_logs": False,
            "log_directory": "logs/",
        }

    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.get("console_logging", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.get("save_logs", True):
        log_dir_str = config.get("log_directory", "logs/")
        log_dir = _PROJECT_ROOT / log_dir_str
        log_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"trinetra_{date_str}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
