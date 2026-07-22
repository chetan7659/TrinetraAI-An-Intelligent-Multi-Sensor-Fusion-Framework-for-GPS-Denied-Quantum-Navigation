"""Custom exception hierarchy for Trinetra-AI."""


class TrinetraError(Exception):
    """Base exception for all Trinetra-AI errors."""

    pass


class ConfigurationError(TrinetraError):
    """Raised when there is an issue with the configuration (e.g., missing keys)."""

    pass


class DatasetError(TrinetraError):
    """Raised when there is an issue loading or parsing datasets."""

    pass


class PreprocessingError(TrinetraError):
    """Raised when an error occurs during data preprocessing."""

    pass


class SensorFusionError(TrinetraError):
    """Raised when sensor fusion algorithms fail to converge or process data."""

    pass


class NavigationError(TrinetraError):
    """Raised when navigation state estimation encounters an error."""

    pass


class ModelError(TrinetraError):
    """Raised during AI model initialization, training, or inference."""

    pass


class EvaluationError(TrinetraError):
    """Raised when benchmarking or evaluation fails."""

    pass
