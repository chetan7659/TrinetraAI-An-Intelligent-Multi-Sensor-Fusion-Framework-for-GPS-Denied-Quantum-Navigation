"""Tests for the centralized logging system."""

import logging

from trinetra.infrastructure.logging import setup_logger
from trinetra.infrastructure.logging.logger import _PROJECT_ROOT


def test_logger_creation():
    """Verify that the logger is created and configured correctly."""
    logger = setup_logger("test_module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"

    # Should have handlers attached (console and file if config allows)
    assert len(logger.handlers) > 0


def test_logger_directory_creation():
    """Verify that the logs directory is created if save_logs is true."""
    # We assume the default logging.yaml has save_logs: true and log_directory: "logs/"
    setup_logger("test_dir_module")

    # Check if the logs directory exists
    log_dir = _PROJECT_ROOT / "logs"
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_logger_does_not_raise_on_log():
    """Verify that logging a message works without errors."""
    logger = setup_logger("test_log_module")

    # These should simply not raise any exceptions
    logger.info("Test dataset preprocessing started.")
    logger.error("Test error message.")
    logger.debug("Test debug message.")
