"""RoNIN Dataset Adapter.

This module provides :class:`RoninAdapter`, the concrete implementation of
:class:`~trinetra.domain.interfaces.dataset_interface.DatasetInterface` for
the RoNIN (Robust Neural Inertial Navigation) dataset.

Responsibilities of this adapter
---------------------------------
* Implement the :class:`DatasetInterface` contract.
* Resolve the dataset root from the configuration system (no hardcoded paths).
* Discover available split directories automatically from the filesystem.
* Discover valid recordings within each split (presence-check only).
* Delegate all filesystem validation to :class:`~.validator.RoninValidator`.
* Register itself in the global :data:`~trinetra.adapters.datasets.registry`.

Non-responsibilities (handled by future components)
----------------------------------------------------
* Opening ``data.hdf5`` files               → future HDF5 reader component
* Parsing ``info.json`` files               → future metadata parser component
* Preprocessing or normalizing sensor data  → future preprocessing pipeline
* Computing any sensor statistics           → future EDA pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trinetra.adapters.datasets.metadata import DatasetMetadata
from trinetra.adapters.datasets.registry import registry
from trinetra.domain.interfaces.dataset_interface import DatasetInterface
from trinetra.infrastructure.config import get_config
from trinetra.shared.exceptions import DatasetError

from .models import Recording
from .validator import DatasetValidationReport, RoninValidator

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_HDF5_FILENAME = "data.hdf5"
_INFO_FILENAME = "info.json"

# These are RoNIN dataset facts, not configuration - they belong here.
_RONIN_NAME = "ronin_dataset"
_RONIN_VERSION = "1.0"
_RONIN_HOMEPAGE = "https://ronin.cs.sfu.ca/"
_RONIN_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
_RONIN_SENSOR_TYPES = ["imu"]
_RONIN_SAMPLING_RATE_HZ = 200.0


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────


class RoninAdapter(DatasetInterface):
    """Adapter for the RoNIN IMU navigation dataset.

    The adapter discovers the dataset structure entirely from the filesystem;
    split names are never hardcoded.  All path resolution is driven by the
    project configuration (``configs/dataset.yaml``).

    Args:
        project_root: Optional override for the project root directory.
            When ``None`` (default), the root is inferred from this file's
            location inside the installed package (the standard approach used
            by the infrastructure config loader).

    Example::

        adapter = RoninAdapter()
        print(adapter.list_splits())
        print(adapter.dataset_statistics())
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root: Path = (
            project_root if project_root is not None else self._resolve_project_root()
        )
        self._config: dict[str, Any] = get_config("dataset").get("ronin", {})
        self._validator = RoninValidator()

    # ── DatasetInterface ──────────────────────────────────────────────────────

    def name(self) -> str:
        """Return the canonical dataset identifier."""
        return _RONIN_NAME

    def version(self) -> str:
        """Return the dataset version string."""
        return _RONIN_VERSION

    def sensor_types(self) -> list[str]:
        """Return the sensor modalities present in the dataset."""
        return list(_RONIN_SENSOR_TYPES)

    def sampling_rate(self) -> float:
        """Return the IMU sampling rate in Hz."""
        return _RONIN_SAMPLING_RATE_HZ

    def license(self) -> str:
        """Return the dataset license string."""
        return _RONIN_LICENSE

    def download_url(self) -> str:
        """Return the URL where the dataset can be obtained."""
        return _RONIN_HOMEPAGE

    def validate(self) -> bool:
        """Return True when the full dataset structure is structurally valid.

        Delegates to :meth:`validate_dataset` and returns the top-level
        ``is_valid`` flag from the report.
        """
        report = self.validate_dataset()
        return report.is_valid

    def load_metadata(self) -> DatasetMetadata:
        """Return the static metadata descriptor for the RoNIN dataset.

        Note:
            This does NOT parse ``info.json`` - that is the responsibility of a
            future metadata parser component.  Only dataset-level (not
            recording-level) metadata is returned here.
        """
        return DatasetMetadata(
            name=self.name(),
            version=self.version(),
            description=(
                "RoNIN: Robust Neural Inertial Navigation in the Wild. "
                "A large-scale IMU dataset collected from smartphones for "
                "pedestrian dead-reckoning research."
            ),
            license=self.license(),
            homepage=self.download_url(),
            sensor_types=self.sensor_types(),
            sampling_rate=self.sampling_rate(),
            ground_truth_available=True,
            supported_tasks=["pedestrian_dead_reckoning", "imu_preintegration"],
        )

    # ── RoNIN-specific public API ─────────────────────────────────────────────

    def homepage(self) -> str:
        """Return the project homepage URL."""
        return _RONIN_HOMEPAGE

    def dataset_root(self) -> Path:
        """Return the resolved absolute path to the RoNIN dataset root.

        The path is read exclusively from ``configs/dataset.yaml`` under the
        ``ronin.dataset_root`` key and resolved relative to the project root.
        No path is ever hardcoded.

        Raises:
            DatasetError: If the ``ronin.dataset_root`` key is missing from
                the configuration.
        """
        raw_path: str | None = self._config.get("dataset_root")
        if raw_path is None:
            raise DatasetError(
                "Configuration key 'ronin.dataset_root' is missing from " "configs/dataset.yaml."
            )
        return (self._project_root / raw_path).resolve()

    def list_splits(self) -> list[str]:
        """Discover and return available split directory names.

        Scans the ``Data/`` subdirectory (configured via ``ronin.data_subdir``)
        and returns the names of all *directories* found there.  No names are
        hardcoded.

        Returns:
            Sorted list of split directory names (e.g.
            ``["seen_subjects_test_set", "train_dataset_1", ...]``).

        Raises:
            DatasetError: If the dataset root or ``Data/`` directory does not
                exist on the filesystem.
        """
        data_dir = self._data_dir()
        if not data_dir.is_dir():
            raise DatasetError(
                f"RoNIN Data directory not found: {data_dir}. "
                "Ensure the dataset has been downloaded to the configured path."
            )
        return sorted(entry.name for entry in data_dir.iterdir() if entry.is_dir())

    def list_recordings(self, split: str) -> list[Recording]:
        """Discover all valid recordings within a given split directory.

        A recording directory is considered valid when it contains *both*
        ``data.hdf5`` and ``info.json``.  Directories missing either file
        are silently ignored (they may be partially downloaded).

        Args:
            split: The split directory name (e.g. ``"train_dataset_1"``).

        Returns:
            Sorted list of :class:`~.models.Recording` objects for every
            valid recording in the split.

        Raises:
            DatasetError: If the requested split directory does not exist.
        """
        split_dir = self._data_dir() / split
        if not split_dir.is_dir():
            raise DatasetError(
                f"Split directory '{split}' not found at: {split_dir}. "
                f"Available splits: {self.list_splits()}"
            )

        recordings: list[Recording] = []
        for entry in sorted(split_dir.iterdir()):
            if not entry.is_dir():
                continue
            hdf5 = entry / _HDF5_FILENAME
            info = entry / _INFO_FILENAME
            if hdf5.is_file() and info.is_file():
                recordings.append(
                    Recording(
                        recording_id=entry.name,
                        split=split,
                        recording_path=entry,
                        hdf5_path=hdf5,
                        info_path=info,
                    )
                )
        return recordings

    def get_recording(self, recording_id: str) -> Recording:
        """Look up a recording by its ID across all splits.

        Args:
            recording_id: The name of the recording directory to find.

        Returns:
            The matching :class:`~.models.Recording`.

        Raises:
            DatasetError: If no recording with the given ID is found in any
                split.
        """
        for split in self.list_splits():
            try:
                for recording in self.list_recordings(split):
                    if recording.recording_id == recording_id:
                        return recording
            except DatasetError:
                continue

        raise DatasetError(
            f"Recording '{recording_id}' was not found in any split of the "
            f"RoNIN dataset at: {self.dataset_root()}"
        )

    def validate_dataset(self) -> DatasetValidationReport:
        """Run the full structural validation and return a detailed report.

        Delegates all filesystem checks to :class:`~.validator.RoninValidator`.

        Returns:
            A :class:`~.validator.DatasetValidationReport` containing
            per-split and per-recording results.
        """
        try:
            splits = self.list_splits()
        except DatasetError:
            splits = []

        return self._validator.validate(
            dataset_root=self.dataset_root(),
            data_subdir=self._data_subdir_name(),
            known_splits=splits,
        )

    def dataset_statistics(self) -> dict[str, Any]:
        """Return high-level structural statistics for the dataset.

        Returns a plain dictionary suitable for logging or display.  No sensor
        data is read; only the filesystem structure is inspected.

        Returns:
            Dictionary with the following keys:

            * ``total_splits`` - number of discovered split directories.
            * ``total_recordings`` - total valid recordings across all splits.
            * ``recordings_per_split`` - mapping of split name to valid count.

        Raises:
            DatasetError: If the ``Data/`` directory cannot be found.
        """
        splits = self.list_splits()
        recordings_per_split: dict[str, int] = {}
        total_recordings = 0

        for split in splits:
            try:
                recs = self.list_recordings(split)
            except DatasetError:
                recs = []
            recordings_per_split[split] = len(recs)
            total_recordings += len(recs)

        return {
            "total_splits": len(splits),
            "total_recordings": total_recordings,
            "recordings_per_split": recordings_per_split,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _data_subdir_name(self) -> str:
        """Return the configured name of the data subdirectory (default: 'Data')."""
        return self._config.get("data_subdir", "Data")

    def _data_dir(self) -> Path:
        """Return the absolute path to the ``Data/`` subdirectory."""
        return self.dataset_root() / self._data_subdir_name()

    @staticmethod
    def _resolve_project_root() -> Path:
        """Resolve the project root from this file's location inside the package.

        File location:
            src/trinetra/adapters/datasets/ronin/adapter.py

        Root is 6 levels up from this file:
            adapter.py → ronin/ → datasets/ → adapters/ → trinetra/ → src/ → <root>
        """
        return Path(__file__).resolve().parents[5]


# ──────────────────────────────────────────────────────────────────────────────
# Auto-registration in the global registry
# ──────────────────────────────────────────────────────────────────────────────

registry.register_dataset(_RONIN_NAME, RoninAdapter)
