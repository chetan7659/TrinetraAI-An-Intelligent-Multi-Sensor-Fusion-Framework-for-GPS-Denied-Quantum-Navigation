"""Immutable data models for the RoNIN dataset adapter.

This module defines pure data structures used throughout the RoNIN adapter.
It has NO external dependencies beyond the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    """Immutable representation of a single RoNIN recording session.

    A recording is a directory that contains exactly two files:
    - ``data.hdf5`` - the raw sensor data file (not opened by this adapter)
    - ``info.json``  - the recording metadata file (not parsed by this adapter)

    Attributes:
        recording_id: Unique name of the recording directory (e.g. "train_dataset_1").
        split: Name of the dataset split this recording belongs to
               (e.g. "train_dataset_1", "seen_subjects_test_set").
        recording_path: Absolute path to the recording directory.
        hdf5_path: Absolute path to the ``data.hdf5`` file inside the recording.
        info_path: Absolute path to the ``info.json`` file inside the recording.

    Notes:
        This class is deliberately *frozen* (immutable). No methods and no
        parsing logic live here - it is a value object only.
    """

    recording_id: str
    split: str
    recording_path: Path
    hdf5_path: Path
    info_path: Path
