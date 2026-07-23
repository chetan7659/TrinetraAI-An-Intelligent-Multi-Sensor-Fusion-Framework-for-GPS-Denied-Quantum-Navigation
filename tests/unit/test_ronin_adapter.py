"""Unit tests for the RoNIN dataset adapter.

All tests use pytest ``tmp_path`` fixtures to construct a synthetic RoNIN
directory tree.  The real RoNIN dataset is NOT required.

Fixture layout (created per-test or per-session as appropriate)::

    <tmp_path>/
    └── datasets/
        └── raw/
            └── ronin/
                └── Data/
                    ├── train_dataset_1/
                    │   ├── rec_001/
                    │   │   ├── data.hdf5   (empty placeholder)
                    │   │   └── info.json   (empty placeholder)
                    │   └── rec_002/
                    │       ├── data.hdf5
                    │       └── info.json
                    ├── seen_subjects_test_set/
                    │   └── rec_010/
                    │       ├── data.hdf5
                    │       └── info.json
                    └── unseen_subjects_test_set/
                        ├── rec_020/              ← valid
                        │   ├── data.hdf5
                        │   └── info.json
                        └── incomplete_rec/       ← missing info.json
                            └── data.hdf5
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trinetra.adapters.datasets.ronin import (
    Recording,
    RoninAdapter,
    RoninValidator,
)
from trinetra.adapters.datasets.ronin.validator import DatasetValidationReport
from trinetra.shared.exceptions import DatasetError

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_recording(parent: Path, name: str, *, complete: bool = True) -> Path:
    """Create a recording directory with placeholder files."""
    rec = parent / name
    rec.mkdir(parents=True)
    (rec / "data.hdf5").touch()
    if complete:
        (rec / "info.json").touch()
    return rec


def _build_dataset(root: Path) -> Path:
    """Construct a synthetic RoNIN dataset tree and return the dataset root."""
    data_dir = root / "datasets" / "raw" / "ronin" / "Data"

    # train_dataset_1 — two complete recordings
    split_train = data_dir / "train_dataset_1"
    _make_recording(split_train, "rec_001")
    _make_recording(split_train, "rec_002")

    # seen_subjects_test_set — one complete recording
    split_seen = data_dir / "seen_subjects_test_set"
    _make_recording(split_seen, "rec_010")

    # unseen_subjects_test_set — one complete, one incomplete
    split_unseen = data_dir / "unseen_subjects_test_set"
    _make_recording(split_unseen, "rec_020")
    _make_recording(split_unseen, "incomplete_rec", complete=False)

    # Also create the Pretrained_Models dir (should be ignored by splits)
    (root / "datasets" / "raw" / "ronin" / "Pretrained_Models").mkdir(parents=True)

    return root / "datasets" / "raw" / "ronin"


def _make_adapter(project_root: Path) -> RoninAdapter:
    """Build a RoninAdapter whose config is overridden to point at tmp_path."""
    # Patch the config loader so the adapter reads from our synthetic structure.
    fake_config = {
        "dataset_root": "datasets/raw/ronin",
        "data_subdir": "Data",
        "excluded_dirs": ["Pretrained_Models"],
    }
    with patch(
        "trinetra.adapters.datasets.ronin.adapter.get_config",
        return_value={"ronin": fake_config},
    ):
        adapter = RoninAdapter(project_root=project_root)
    # Re-apply the fake config post-init (it is stored in self._config)
    adapter._config = fake_config
    return adapter


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """Build the synthetic dataset tree and return the dataset root."""
    return _build_dataset(tmp_path)


@pytest.fixture()
def adapter(tmp_path: Path, dataset_root: Path) -> RoninAdapter:
    """Return a RoninAdapter pointed at the synthetic tmp_path tree."""
    return _make_adapter(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Static metadata tests
# ──────────────────────────────────────────────────────────────────────────────


class TestStaticMetadata:
    def test_name(self, adapter: RoninAdapter) -> None:
        assert adapter.name() == "ronin_dataset"

    def test_version(self, adapter: RoninAdapter) -> None:
        assert isinstance(adapter.version(), str)
        assert len(adapter.version()) > 0

    def test_sensor_types(self, adapter: RoninAdapter) -> None:
        assert "imu" in adapter.sensor_types()

    def test_sampling_rate(self, adapter: RoninAdapter) -> None:
        assert adapter.sampling_rate() == 200.0

    def test_license_non_empty(self, adapter: RoninAdapter) -> None:
        assert len(adapter.license()) > 0

    def test_download_url_is_https(self, adapter: RoninAdapter) -> None:
        assert adapter.download_url().startswith("https://")

    def test_load_metadata_returns_correct_type(self, adapter: RoninAdapter) -> None:
        from trinetra.adapters.datasets.metadata import DatasetMetadata

        meta = adapter.load_metadata()
        assert isinstance(meta, DatasetMetadata)
        assert meta.ground_truth_available is True
        assert "pedestrian_dead_reckoning" in meta.supported_tasks


# ──────────────────────────────────────────────────────────────────────────────
# Split discovery tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSplitDiscovery:
    def test_list_splits_discovers_three_splits(self, adapter: RoninAdapter) -> None:
        splits = adapter.list_splits()
        assert len(splits) == 3

    def test_list_splits_sorted_alphabetically(self, adapter: RoninAdapter) -> None:
        splits = adapter.list_splits()
        assert splits == sorted(splits)

    def test_list_splits_contains_expected_splits(self, adapter: RoninAdapter) -> None:
        splits = adapter.list_splits()
        assert "train_dataset_1" in splits
        assert "seen_subjects_test_set" in splits
        assert "unseen_subjects_test_set" in splits

    def test_list_splits_missing_data_dir_raises(self, tmp_path: Path) -> None:
        """Adapter should raise DatasetError when Data/ does not exist."""
        # Build dataset root without the Data/ subdirectory
        (tmp_path / "datasets" / "raw" / "ronin").mkdir(parents=True)
        adapter = _make_adapter(tmp_path)
        with pytest.raises(DatasetError):
            adapter.list_splits()


# ──────────────────────────────────────────────────────────────────────────────
# Recording discovery tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRecordingDiscovery:
    def test_list_recordings_returns_recording_objects(self, adapter: RoninAdapter) -> None:
        recs = adapter.list_recordings("train_dataset_1")
        assert all(isinstance(r, Recording) for r in recs)

    def test_list_recordings_train_split_has_two_recordings(self, adapter: RoninAdapter) -> None:
        recs = adapter.list_recordings("train_dataset_1")
        assert len(recs) == 2

    def test_list_recordings_seen_split_has_one_recording(self, adapter: RoninAdapter) -> None:
        recs = adapter.list_recordings("seen_subjects_test_set")
        assert len(recs) == 1

    def test_list_recordings_filters_incomplete_recordings(self, adapter: RoninAdapter) -> None:
        """Only 1 of 2 directories in unseen split has both required files."""
        recs = adapter.list_recordings("unseen_subjects_test_set")
        assert len(recs) == 1
        assert recs[0].recording_id == "rec_020"

    def test_list_recordings_recording_fields_are_correct(self, adapter: RoninAdapter) -> None:
        recs = adapter.list_recordings("train_dataset_1")
        rec = recs[0]
        assert rec.split == "train_dataset_1"
        assert rec.hdf5_path.name == "data.hdf5"
        assert rec.info_path.name == "info.json"
        assert rec.recording_path.is_dir()

    def test_list_recordings_hdf5_and_info_paths_exist(self, adapter: RoninAdapter) -> None:
        recs = adapter.list_recordings("train_dataset_1")
        for rec in recs:
            assert rec.hdf5_path.exists()
            assert rec.info_path.exists()

    def test_list_recordings_invalid_split_raises(self, adapter: RoninAdapter) -> None:
        with pytest.raises(DatasetError, match="nonexistent_split"):
            adapter.list_recordings("nonexistent_split")

    def test_recording_is_immutable(self, adapter: RoninAdapter) -> None:
        """Recording is a frozen dataclass; attribute assignment must fail."""
        rec = adapter.list_recordings("seen_subjects_test_set")[0]
        with pytest.raises((AttributeError, TypeError)):
            rec.recording_id = "mutated"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# Recording lookup tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRecordingLookup:
    def test_get_recording_returns_correct_recording(self, adapter: RoninAdapter) -> None:
        rec = adapter.get_recording("rec_001")
        assert rec.recording_id == "rec_001"

    def test_get_recording_finds_across_splits(self, adapter: RoninAdapter) -> None:
        # rec_010 lives in seen_subjects_test_set
        rec = adapter.get_recording("rec_010")
        assert rec.split == "seen_subjects_test_set"

    def test_get_recording_not_found_raises_dataset_error(self, adapter: RoninAdapter) -> None:
        with pytest.raises(DatasetError, match="does_not_exist"):
            adapter.get_recording("does_not_exist")

    def test_get_recording_incomplete_not_findable(self, adapter: RoninAdapter) -> None:
        """Incomplete recordings are excluded from list_recordings; get_recording
        must therefore also not find them."""
        with pytest.raises(DatasetError):
            adapter.get_recording("incomplete_rec")


# ──────────────────────────────────────────────────────────────────────────────
# Statistics tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasetStatistics:
    def test_statistics_returns_dict(self, adapter: RoninAdapter) -> None:
        stats = adapter.dataset_statistics()
        assert isinstance(stats, dict)

    def test_statistics_total_splits(self, adapter: RoninAdapter) -> None:
        stats = adapter.dataset_statistics()
        assert stats["total_splits"] == 3

    def test_statistics_total_recordings(self, adapter: RoninAdapter) -> None:
        # train=2, seen=1, unseen=1 (incomplete excluded) → 4 total
        stats = adapter.dataset_statistics()
        assert stats["total_recordings"] == 4

    def test_statistics_recordings_per_split_keys(self, adapter: RoninAdapter) -> None:
        stats = adapter.dataset_statistics()
        per_split = stats["recordings_per_split"]
        assert "train_dataset_1" in per_split
        assert "seen_subjects_test_set" in per_split
        assert "unseen_subjects_test_set" in per_split

    def test_statistics_recordings_per_split_values(self, adapter: RoninAdapter) -> None:
        stats = adapter.dataset_statistics()
        per_split = stats["recordings_per_split"]
        assert per_split["train_dataset_1"] == 2
        assert per_split["seen_subjects_test_set"] == 1
        assert per_split["unseen_subjects_test_set"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_validate_returns_true_for_complete_splits(self, adapter: RoninAdapter) -> None:
        # The unseen split has an incomplete recording, so overall is invalid.
        # We test a custom validator call against only the clean splits.
        validator = RoninValidator()
        report = validator.validate(
            dataset_root=adapter.dataset_root(),
            data_subdir="Data",
            known_splits=["train_dataset_1", "seen_subjects_test_set"],
        )
        assert report.is_valid is True

    def test_validate_returns_false_when_incomplete_recording_present(
        self, adapter: RoninAdapter
    ) -> None:
        report: DatasetValidationReport = adapter.validate_dataset()
        # unseen_subjects_test_set has an incomplete recording → invalid
        assert report.is_valid is False

    def test_validate_dataset_root_exists(self, adapter: RoninAdapter) -> None:
        report = adapter.validate_dataset()
        assert report.dataset_root_exists is True

    def test_validate_data_dir_exists(self, adapter: RoninAdapter) -> None:
        report = adapter.validate_dataset()
        assert report.data_dir_exists is True

    def test_validate_split_count_in_report(self, adapter: RoninAdapter) -> None:
        report = adapter.validate_dataset()
        assert len(report.split_results) == 3

    def test_validate_as_dict_contains_expected_keys(self, adapter: RoninAdapter) -> None:
        report = adapter.validate_dataset()
        d = report.as_dict()
        assert "dataset_root" in d
        assert "is_valid" in d
        assert "splits" in d

    def test_validate_missing_root_marks_invalid(self, tmp_path: Path) -> None:
        """Validation report is invalid when the dataset root is absent."""
        adapter = _make_adapter(tmp_path)  # no dataset tree built
        report = adapter.validate_dataset()
        assert report.dataset_root_exists is False
        assert report.is_valid is False

    def test_validate_method_delegates_correctly(self, adapter: RoninAdapter) -> None:
        """adapter.validate() must return a bool consistent with the report."""
        report = adapter.validate_dataset()
        assert adapter.validate() == report.is_valid


# ──────────────────────────────────────────────────────────────────────────────
# Dataset root configuration tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasetRoot:
    def test_dataset_root_returns_path_object(self, adapter: RoninAdapter) -> None:
        assert isinstance(adapter.dataset_root(), Path)

    def test_dataset_root_is_absolute(self, adapter: RoninAdapter) -> None:
        assert adapter.dataset_root().is_absolute()

    def test_dataset_root_points_inside_tmp(self, adapter: RoninAdapter, tmp_path: Path) -> None:
        root = adapter.dataset_root()
        # The root must be a descendant of tmp_path
        assert str(root).startswith(str(tmp_path))

    def test_missing_config_key_raises_dataset_error(self, tmp_path: Path) -> None:
        """If 'ronin.dataset_root' is absent from config, raise DatasetError."""
        fake_config: dict = {}  # intentionally empty ronin config
        with patch(
            "trinetra.adapters.datasets.ronin.adapter.get_config",
            return_value={"ronin": fake_config},
        ):
            adapter = RoninAdapter(project_root=tmp_path)
        adapter._config = fake_config
        with pytest.raises(DatasetError, match="dataset_root"):
            adapter.dataset_root()
