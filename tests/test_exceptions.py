"""Tests for the exception hierarchy."""

import pytest

from trinetra.shared.exceptions import (
    ConfigurationError,
    DatasetError,
    EvaluationError,
    ModelError,
    NavigationError,
    PreprocessingError,
    SensorFusionError,
    TrinetraError,
)


def test_exception_hierarchy():
    """Verify that all custom exceptions inherit from the base TrinetraError."""
    exceptions = [
        ConfigurationError,
        DatasetError,
        PreprocessingError,
        SensorFusionError,
        NavigationError,
        ModelError,
        EvaluationError,
    ]

    for exc in exceptions:
        assert issubclass(exc, TrinetraError)
        assert issubclass(exc, Exception)


def test_exception_raising():
    """Verify that exceptions can be raised and caught correctly."""
    with pytest.raises(DatasetError) as exc_info:
        raise DatasetError("Dataset directory not found.")

    assert "Dataset directory not found." in str(exc_info.value)

    # Verify it can be caught as the base error too
    with pytest.raises(TrinetraError):
        raise ConfigurationError("Missing key.")
