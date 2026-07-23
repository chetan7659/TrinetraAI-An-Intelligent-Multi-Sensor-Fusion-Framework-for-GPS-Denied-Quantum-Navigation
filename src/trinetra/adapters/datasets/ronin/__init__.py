"""RoNIN dataset adapter package.

Exposes the primary public API for the RoNIN adapter so that consumers
only need to import from this package.

Example::

    from trinetra.adapters.datasets.ronin import RoninAdapter, RoninMetadataLoader

    adapter  = RoninAdapter()
    recording = adapter.get_recording("a000_1")

    loader   = RoninMetadataLoader()
    metadata = loader.load(recording)
    print(metadata.device)
    print(metadata.length)
"""

from .adapter import RoninAdapter
from .metadata_loader import RoninMetadataLoader
from .metadata_models import (
    ImuCalibration,
    OrientationCalibration,
    OrientationErrors,
    RoninRecordingMetadata,
    TimeSynchronization,
)
from .models import Recording
from .validator import (
    DatasetValidationReport,
    RecordingValidationResult,
    RoninValidator,
    SplitValidationResult,
)

__all__ = [
    "DatasetValidationReport",
    "ImuCalibration",
    "OrientationCalibration",
    "OrientationErrors",
    "Recording",
    "RecordingValidationResult",
    "RoninAdapter",
    "RoninMetadataLoader",
    "RoninRecordingMetadata",
    "RoninValidator",
    "SplitValidationResult",
    "TimeSynchronization",
]
