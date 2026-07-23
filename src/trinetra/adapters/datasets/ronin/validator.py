"""Filesystem validation logic for the RoNIN dataset.

The ``RoninValidator`` class encapsulates all filesystem checks required to
verify that a RoNIN dataset tree is structurally correct before the adapter
attempts to use it.  All validation logic lives here so that ``adapter.py``
stays focused on dataset discovery and coordination.

This module depends only on the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Result structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RecordingValidationResult:
    """Validation result for a single recording directory.

    Attributes:
        recording_id: Name of the recording directory.
        split: The split this recording belongs to.
        recording_path: Absolute path to the recording directory.
        has_hdf5: Whether ``data.hdf5`` is present.
        has_info: Whether ``info.json`` is present.
        is_valid: True only when both required files are present.
    """

    recording_id: str
    split: str
    recording_path: Path
    has_hdf5: bool
    has_info: bool

    @property
    def is_valid(self) -> bool:
        """Return True when both required files are present."""
        return self.has_hdf5 and self.has_info

    @property
    def missing_files(self) -> list[str]:
        """Return a list of required file names that are absent."""
        missing: list[str] = []
        if not self.has_hdf5:
            missing.append("data.hdf5")
        if not self.has_info:
            missing.append("info.json")
        return missing


@dataclass
class SplitValidationResult:
    """Validation result for a single split directory.

    Attributes:
        split_name: Name of the split directory.
        split_path: Absolute path to the split directory.
        exists: Whether the split directory exists on disk.
        recording_results: Validation results for every recording inside.
    """

    split_name: str
    split_path: Path
    exists: bool
    recording_results: list[RecordingValidationResult] = field(default_factory=list)

    @property
    def total_recordings(self) -> int:
        """Total number of discovered recording directories in this split."""
        return len(self.recording_results)

    @property
    def valid_recordings(self) -> int:
        """Number of recordings that contain both required files."""
        return sum(1 for r in self.recording_results if r.is_valid)

    @property
    def invalid_recordings(self) -> int:
        """Number of recordings that are missing one or both required files."""
        return self.total_recordings - self.valid_recordings

    @property
    def is_valid(self) -> bool:
        """A split is valid when it exists and every recording inside it is valid."""
        return self.exists and self.invalid_recordings == 0


@dataclass
class DatasetValidationReport:
    """Top-level validation report for the entire RoNIN dataset tree.

    Attributes:
        dataset_root: Resolved absolute path that was validated.
        dataset_root_exists: Whether the configured dataset root directory exists.
        data_dir_exists: Whether the ``Data/`` subdirectory exists.
        split_results: Per-split validation results.
    """

    dataset_root: Path
    dataset_root_exists: bool
    data_dir_exists: bool
    split_results: list[SplitValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Dataset is valid when the root, Data/ dir, and all splits are valid."""
        return (
            self.dataset_root_exists
            and self.data_dir_exists
            and all(s.is_valid for s in self.split_results)
        )

    @property
    def total_valid_recordings(self) -> int:
        """Total number of valid recordings across all splits."""
        return sum(s.valid_recordings for s in self.split_results)

    @property
    def total_invalid_recordings(self) -> int:
        """Total number of invalid (incomplete) recordings across all splits."""
        return sum(s.invalid_recordings for s in self.split_results)

    def as_dict(self) -> dict:
        """Return the validation report as a plain dictionary for logging/display."""
        return {
            "dataset_root": str(self.dataset_root),
            "dataset_root_exists": self.dataset_root_exists,
            "data_dir_exists": self.data_dir_exists,
            "is_valid": self.is_valid,
            "total_valid_recordings": self.total_valid_recordings,
            "total_invalid_recordings": self.total_invalid_recordings,
            "splits": [
                {
                    "split": s.split_name,
                    "exists": s.exists,
                    "total_recordings": s.total_recordings,
                    "valid_recordings": s.valid_recordings,
                    "invalid_recordings": s.invalid_recordings,
                    "recordings": [
                        {
                            "recording_id": r.recording_id,
                            "is_valid": r.is_valid,
                            "missing_files": r.missing_files,
                        }
                        for r in s.recording_results
                    ],
                }
                for s in self.split_results
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────

_HDF5_FILENAME = "data.hdf5"
_INFO_FILENAME = "info.json"


class RoninValidator:
    """Filesystem validator for the RoNIN dataset directory structure.

    Responsibilities:
    - Verify that the dataset root and ``Data/`` sub-directory exist.
    - Discover every split directory inside ``Data/``.
    - For each split, verify every recording directory contains
      ``data.hdf5`` and ``info.json``.

    Non-responsibilities:
    - Does NOT open any file.
    - Does NOT parse HDF5 or JSON.
    - Does NOT check checksums or file sizes.
    """

    def validate(
        self,
        dataset_root: Path,
        data_subdir: str,
        known_splits: list[str],
    ) -> DatasetValidationReport:
        """Run the full structural validation and return a report.

        Args:
            dataset_root: Absolute path to the RoNIN dataset root directory.
            data_subdir: Name of the subdirectory that holds split folders
                         (typically ``"Data"``).
            known_splits: List of split directory names to validate.  Passing
                          an empty list skips per-split checks.

        Returns:
            A :class:`DatasetValidationReport` with complete per-split and
            per-recording results.
        """
        root_exists = dataset_root.is_dir()
        data_dir = dataset_root / data_subdir
        data_dir_exists = data_dir.is_dir()

        report = DatasetValidationReport(
            dataset_root=dataset_root,
            dataset_root_exists=root_exists,
            data_dir_exists=data_dir_exists,
        )

        if not data_dir_exists:
            return report

        for split_name in known_splits:
            split_path = data_dir / split_name
            split_result = self._validate_split(split_name, split_path)
            report.split_results.append(split_result)

        return report

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_split(self, split_name: str, split_path: Path) -> SplitValidationResult:
        """Validate a single split directory and all recordings inside it."""
        split_exists = split_path.is_dir()
        result = SplitValidationResult(
            split_name=split_name,
            split_path=split_path,
            exists=split_exists,
        )

        if not split_exists:
            return result

        for entry in sorted(split_path.iterdir()):
            if entry.is_dir():
                rec_result = self._validate_recording(entry, split_name)
                result.recording_results.append(rec_result)

        return result

    @staticmethod
    def _validate_recording(recording_path: Path, split_name: str) -> RecordingValidationResult:
        """Check whether a recording directory contains the two required files."""
        return RecordingValidationResult(
            recording_id=recording_path.name,
            split=split_name,
            recording_path=recording_path,
            has_hdf5=(recording_path / _HDF5_FILENAME).is_file(),
            has_info=(recording_path / _INFO_FILENAME).is_file(),
        )
