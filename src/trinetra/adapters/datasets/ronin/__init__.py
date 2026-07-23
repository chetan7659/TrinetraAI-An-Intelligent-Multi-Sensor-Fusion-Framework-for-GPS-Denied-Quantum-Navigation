"""RoNIN dataset adapter package.

Exposes the primary public API for the RoNIN adapter so that consumers
only need to import from this package.

Example::

    from trinetra.adapters.datasets.ronin import (
        RoninAdapter,
        RoninMetadataLoader,
        RoninHDF5Reader,
    )

    adapter   = RoninAdapter()
    recording = adapter.get_recording("a000_1")

    loader    = RoninMetadataLoader()
    metadata  = loader.load(recording)

    reader    = RoninHDF5Reader()
    for frame in reader.read(recording, metadata):
        print(frame.timestamp, frame.acce)
"""

from .adapter import RoninAdapter
from .hdf5_reader import RoninHDF5Reader
from .metadata_loader import RoninMetadataLoader
from .metadata_models import (
    ImuCalibration,
    OrientationCalibration,
    OrientationErrors,
    RoninRecordingMetadata,
    TimeSynchronization,
)
from .models import Recording
from .raw_frames import RoninRawFrame
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
    "RoninHDF5Reader",
    "RoninMetadataLoader",
    "RoninRawFrame",
    "RoninRecordingMetadata",
    "RoninValidator",
    "SplitValidationResult",
    "TimeSynchronization",
]
