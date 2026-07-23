"""RoNIN HDF5 file decoder.

This module provides :class:`RoninHDF5Reader`, which opens a RoNIN
``data.hdf5`` file and decodes the ``synced`` group into a lazy iterator
of :class:`~.raw_frames.RoninRawFrame` objects.

Responsibilities
----------------
* Open ``recording.hdf5_path`` using ``h5py``.
* Validate that the ``synced`` group and all required datasets exist.
* Validate that all datasets share a consistent length.
* Apply the ``start_frame`` offset (from ``RoninRecordingMetadata``) so
  that only post-calibration frames are emitted.
* Yield :class:`~.raw_frames.RoninRawFrame` objects one at a time (lazy
  iteration) to support long recordings without loading everything into RAM.

Non-responsibilities
--------------------
* Converting ``RoninRawFrame`` to domain ``SensorRecord``
  → :class:`~.canonical_mapper.RoninCanonicalMapper` (M1.5)
* Normalising or calibrating sensor values → preprocessing pipeline (M2)
* Re-reading ``info.json`` → :class:`~.metadata_loader.RoninMetadataLoader`

Dependency rule
---------------
This module MUST NOT import anything from ``trinetra.domain``.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import h5py
import numpy as np

from trinetra.shared.exceptions import DatasetError

from .metadata_models import RoninRecordingMetadata
from .models import Recording
from .raw_frames import RoninRawFrame

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_SYNCED_GROUP = "synced"

# Required datasets within the synced group (verified against real files).
_REQUIRED_DATASETS = (
    "time",
    "acce",
    "gyro",
    "gyro_uncalib",
    "linacce",
    "grav",
    "game_rv",
    "rv",
    "magnet",
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_hdf5_file(path: Path, recording_id: str) -> None:
    """Raise :exc:`DatasetError` if the HDF5 file does not exist or is unreadable."""
    if not path.exists():
        raise DatasetError(f"[{recording_id}] HDF5 file not found: {path}")
    if not path.is_file():
        raise DatasetError(f"[{recording_id}] HDF5 path is not a file: {path}")


def _validate_synced_group(hdf5_file: h5py.File, recording_id: str) -> h5py.Group:
    """Return the ``synced`` group or raise :exc:`DatasetError` if absent."""
    if _SYNCED_GROUP not in hdf5_file:
        raise DatasetError(
            f"[{recording_id}] HDF5 file is missing the required "
            f"'{_SYNCED_GROUP}' group.  Found top-level keys: "
            f"{sorted(hdf5_file.keys())}"
        )
    return hdf5_file[_SYNCED_GROUP]


def _validate_required_datasets(synced: h5py.Group, recording_id: str) -> None:
    """Raise :exc:`DatasetError` if any required dataset is absent from ``synced``."""
    missing = [k for k in _REQUIRED_DATASETS if k not in synced]
    if missing:
        raise DatasetError(
            f"[{recording_id}] '{_SYNCED_GROUP}' group is missing required "
            f"dataset(s): {missing}.  Present keys: {sorted(synced.keys())}"
        )


def _validate_consistent_lengths(synced: h5py.Group, recording_id: str) -> int:
    """Verify all synced datasets share the same first-axis length.

    Returns the common length (number of frames in the file).

    Raises:
        DatasetError: If any dataset has a different first-axis length.
    """
    lengths: dict[str, int] = {k: synced[k].shape[0] for k in _REQUIRED_DATASETS}
    unique = set(lengths.values())
    if len(unique) != 1:
        raise DatasetError(
            f"[{recording_id}] Inconsistent dataset lengths in " f"'{_SYNCED_GROUP}': {lengths}"
        )
    return unique.pop()


def _validate_start_frame(start_frame: int, total_frames: int, recording_id: str) -> None:
    """Raise :exc:`DatasetError` if ``start_frame`` is out of range."""
    if start_frame < 0 or start_frame >= total_frames:
        raise DatasetError(
            f"[{recording_id}] start_frame={start_frame} is out of range "
            f"for a recording with {total_frames} total frames."
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class RoninHDF5Reader:
    """Decodes a RoNIN ``data.hdf5`` file into an iterable of
    :class:`~.raw_frames.RoninRawFrame` objects.

    This class is **stateless** — a single instance may be reused to read
    multiple recordings sequentially.

    Design contract
    ---------------
    * Opens ``recording.hdf5_path`` **only**.
    * Reads **only** the ``synced`` group (the time-synchronised 200 Hz
      stream).  The ``raw`` and ``pose`` groups are never accessed.
    * Does **not** import from ``trinetra.domain``.
    * Does **not** re-parse ``info.json`` — calibration metadata is accepted
      as an optional ``RoninRecordingMetadata`` argument.

    Example::

        adapter  = RoninAdapter()
        recording = adapter.get_recording("a000_1")

        loader   = RoninMetadataLoader()
        metadata = loader.load(recording)

        reader   = RoninHDF5Reader()
        for frame in reader.read(recording, metadata):
            print(frame.timestamp, frame.acce)
    """

    def read(
        self,
        recording: Recording,
        metadata: RoninRecordingMetadata | None = None,
    ) -> Generator[RoninRawFrame, None, None]:
        """Decode ``data.hdf5`` and yield one :class:`~.raw_frames.RoninRawFrame`
        per synchronized time step.

        Args:
            recording: A :class:`~.models.Recording` returned by
                :class:`~.adapter.RoninAdapter`.
            metadata: Optional :class:`~.metadata_models.RoninRecordingMetadata`
                loaded by :class:`~.metadata_loader.RoninMetadataLoader`.
                When provided, ``metadata.start_frame`` is used to skip
                pre-calibration frames.  When ``None``, all frames are emitted
                from index 0 (useful for inspection and testing).

        Yields:
            :class:`~.raw_frames.RoninRawFrame` — one per post-calibration
            time step.  Frames are yielded in chronological order.

        Raises:
            DatasetError: If:

                * ``recording.hdf5_path`` does not exist,
                * the file cannot be opened as a valid HDF5 archive,
                * the ``synced`` group is absent,
                * any required dataset is missing,
                * dataset lengths are inconsistent,
                * ``start_frame`` (from metadata) is out of range.
        """
        _validate_hdf5_file(recording.hdf5_path, recording.recording_id)

        try:
            hdf5_file = h5py.File(recording.hdf5_path, "r")
        except OSError as exc:
            raise DatasetError(
                f"[{recording.recording_id}] Failed to open HDF5 file "
                f"'{recording.hdf5_path}': {exc}"
            ) from exc

        with hdf5_file:
            yield from self._decode(hdf5_file, recording, metadata)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _decode(
        hdf5_file: h5py.File,
        recording: Recording,
        metadata: RoninRecordingMetadata | None,
    ) -> Generator[RoninRawFrame, None, None]:
        """Internal: validate and lazily decode the synced group."""
        rid = recording.recording_id

        synced = _validate_synced_group(hdf5_file, rid)
        _validate_required_datasets(synced, rid)
        total_frames = _validate_consistent_lengths(synced, rid)

        # Determine the starting frame offset.
        start_frame: int = metadata.start_frame if metadata is not None else 0
        _validate_start_frame(start_frame, total_frames, rid)

        # Load the entire synced group into memory as NumPy arrays.
        # h5py datasets support numpy-style slicing; we slice from start_frame
        # to avoid pulling in pre-calibration frames we will never use.
        # Each slice returns a NumPy ndarray; no unnecessary copies are made.
        time_arr: np.ndarray = synced["time"][start_frame:]
        acce_arr: np.ndarray = synced["acce"][start_frame:]
        gyro_arr: np.ndarray = synced["gyro"][start_frame:]
        gyro_u_arr: np.ndarray = synced["gyro_uncalib"][start_frame:]
        linacce_arr: np.ndarray = synced["linacce"][start_frame:]
        grav_arr: np.ndarray = synced["grav"][start_frame:]
        game_rv_arr: np.ndarray = synced["game_rv"][start_frame:]
        rv_arr: np.ndarray = synced["rv"][start_frame:]
        magnet_arr: np.ndarray = synced["magnet"][start_frame:]

        n_frames = len(time_arr)

        for i in range(n_frames):
            yield RoninRawFrame(
                frame_index=i,
                timestamp=float(time_arr[i]),
                acce=(
                    float(acce_arr[i, 0]),
                    float(acce_arr[i, 1]),
                    float(acce_arr[i, 2]),
                ),
                gyro=(
                    float(gyro_arr[i, 0]),
                    float(gyro_arr[i, 1]),
                    float(gyro_arr[i, 2]),
                ),
                gyro_uncalib=(
                    float(gyro_u_arr[i, 0]),
                    float(gyro_u_arr[i, 1]),
                    float(gyro_u_arr[i, 2]),
                ),
                linacce=(
                    float(linacce_arr[i, 0]),
                    float(linacce_arr[i, 1]),
                    float(linacce_arr[i, 2]),
                ),
                grav=(
                    float(grav_arr[i, 0]),
                    float(grav_arr[i, 1]),
                    float(grav_arr[i, 2]),
                ),
                game_rv=(
                    float(game_rv_arr[i, 0]),
                    float(game_rv_arr[i, 1]),
                    float(game_rv_arr[i, 2]),
                    float(game_rv_arr[i, 3]),
                ),
                rv=(
                    float(rv_arr[i, 0]),
                    float(rv_arr[i, 1]),
                    float(rv_arr[i, 2]),
                    float(rv_arr[i, 3]),
                ),
                magnet=(
                    float(magnet_arr[i, 0]),
                    float(magnet_arr[i, 1]),
                    float(magnet_arr[i, 2]),
                ),
            )
