"""Unit tests for RoninMetadataLoader and RoninRecordingMetadata.

All tests use pytest ``tmp_path`` fixtures to construct synthetic info.json files.
The real RoNIN dataset is NOT required.

Synthetic schema used (mirrors the real info.json exactly)::

    {
        "length":                  338.045,
        "date":                    "01/08/19",
        "device":                  "asus5",
        "type":                    "annotated",
        "start_frame":             5269,
        "imu_init_gyro_bias":      [0.0105, 0.0034, 0.0098],
        "imu_end_gyro_bias":       [0.0104, 0.0034, 0.0096],
        "imu_acce_bias":           [-0.096, -0.013, 0.023],
        "imu_acce_scale":          [0.9985, 0.9970, 0.9979],
        "imu_reference_time":      20994527127022.0,
        "tango_reference_time":    18224109457042.0,
        "imu_time_offset":         0.028087,
        "align_tango_to_body":     [0.5465, 0.5052, 0.4728, -0.4715],
        "start_calibration":       [0.0, 0.9999, -0.0011, 0.0],
        "end_calibration":         [0.0, 0.9999, 0.0068, 0.0],
        "gyro_integration_error":  3.769,
        "grv_ori_error":           3.519,
        "ekf_ori_error":           2.534
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trinetra.adapters.datasets.ronin.metadata_loader import RoninMetadataLoader
from trinetra.adapters.datasets.ronin.metadata_models import (
    ImuCalibration,
    OrientationCalibration,
    OrientationErrors,
    RoninRecordingMetadata,
    TimeSynchronization,
)
from trinetra.adapters.datasets.ronin.models import Recording
from trinetra.shared.exceptions import DatasetError

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures and helpers
# ──────────────────────────────────────────────────────────────────────────────

VALID_INFO: dict[str, Any] = {
    "length": 338.045,
    "date": "01/08/19",
    "device": "asus5",
    "type": "annotated",
    "start_frame": 5269,
    "imu_init_gyro_bias": [0.0105, 0.0034, 0.0098],
    "imu_end_gyro_bias": [0.0104, 0.0034, 0.0096],
    "imu_acce_bias": [-0.096, -0.013, 0.023],
    "imu_acce_scale": [0.9985, 0.9970, 0.9979],
    "imu_reference_time": 20994527127022.0,
    "tango_reference_time": 18224109457042.0,
    "imu_time_offset": 0.028087,
    "align_tango_to_body": [0.5465, 0.5052, 0.4728, -0.4715],
    "start_calibration": [0.0, 0.9999, -0.0011, 0.0],
    "end_calibration": [0.0, 0.9999, 0.0068, 0.0],
    "gyro_integration_error": 3.769,
    "grv_ori_error": 3.519,
    "ekf_ori_error": 2.534,
}


def _make_recording(tmp_path: Path, info_data: Any = None, *, write_file: bool = True) -> Recording:
    """Build a minimal Recording pointing at a synthetic directory."""
    rec_dir = tmp_path / "a000_1"
    rec_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = rec_dir / "data.hdf5"
    hdf5_path.touch()
    info_path = rec_dir / "info.json"

    if write_file:
        info_path.write_text(
            json.dumps(info_data if info_data is not None else VALID_INFO), encoding="utf-8"
        )

    return Recording(
        recording_id="a000_1",
        split="train_dataset_1",
        recording_path=rec_dir,
        hdf5_path=hdf5_path,
        info_path=info_path,
    )


@pytest.fixture()
def loader() -> RoninMetadataLoader:
    return RoninMetadataLoader()


@pytest.fixture()
def valid_recording(tmp_path: Path) -> Recording:
    return _make_recording(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Valid metadata — correct parsing
# ──────────────────────────────────────────────────────────────────────────────


class TestValidMetadata:
    def test_load_returns_correct_type(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta, RoninRecordingMetadata)

    def test_recording_id_injected_from_recording(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.recording_id == "a000_1"

    def test_split_injected_from_recording(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.split == "train_dataset_1"

    def test_length_parsed_correctly(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.length == pytest.approx(338.045)

    def test_date_parsed_correctly(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.date == "01/08/19"

    def test_device_parsed_correctly(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.device == "asus5"

    def test_recording_type_parsed(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.recording_type == "annotated"

    def test_start_frame_is_int(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta.start_frame, int)
        assert meta.start_frame == 5269

    def test_calibration_is_correct_type(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta.calibration, ImuCalibration)

    def test_calibration_imu_acce_scale(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert len(meta.calibration.imu_acce_scale) == 3
        assert meta.calibration.imu_acce_scale[0] == pytest.approx(0.9985)

    def test_calibration_gyro_bias_tuples(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta.calibration.imu_init_gyro_bias, tuple)
        assert isinstance(meta.calibration.imu_end_gyro_bias, tuple)

    def test_time_sync_is_correct_type(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta.time_sync, TimeSynchronization)

    def test_time_sync_imu_reference_time(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert meta.time_sync.imu_reference_time == pytest.approx(20994527127022.0)

    def test_orientation_is_correct_type(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta.orientation, OrientationCalibration)

    def test_orientation_align_tango_to_body_len(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert len(meta.orientation.align_tango_to_body) == 4

    def test_errors_is_correct_type(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        assert isinstance(meta.errors, OrientationErrors)

    def test_errors_values(self, loader: RoninMetadataLoader, valid_recording: Recording) -> None:
        meta = loader.load(valid_recording)
        assert meta.errors.gyro_integration_error == pytest.approx(3.769)
        assert meta.errors.grv_ori_error == pytest.approx(3.519)
        assert meta.errors.ekf_ori_error == pytest.approx(2.534)

    def test_int_length_field_coerced_to_float(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        """JSON integers in numeric fields should be accepted and cast to float."""
        data = dict(VALID_INFO)
        data["length"] = 338  # int in JSON
        recording = _make_recording(tmp_path, data)
        meta = loader.load(recording)
        assert isinstance(meta.length, float)
        assert meta.length == pytest.approx(338.0)

    def test_loader_is_reusable_across_recordings(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        """Loader must be stateless; loading two different recordings is safe."""
        rec_a = _make_recording(tmp_path / "a", VALID_INFO)
        data_b = dict(VALID_INFO, device="samsung1")
        rec_b = _make_recording(tmp_path / "b", data_b)

        meta_a = loader.load(rec_a)
        meta_b = loader.load(rec_b)
        assert meta_a.device == "asus5"
        assert meta_b.device == "samsung1"


# ──────────────────────────────────────────────────────────────────────────────
# Immutability of the metadata model
# ──────────────────────────────────────────────────────────────────────────────


class TestImmutability:
    def test_metadata_top_level_is_immutable(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        with pytest.raises((AttributeError, TypeError)):
            meta.device = "mutated"  # type: ignore[misc]

    def test_calibration_is_immutable(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        with pytest.raises((AttributeError, TypeError)):
            meta.calibration.imu_acce_bias = (0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_time_sync_is_immutable(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        with pytest.raises((AttributeError, TypeError)):
            meta.time_sync.imu_time_offset = 0.0  # type: ignore[misc]

    def test_orientation_is_immutable(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        with pytest.raises((AttributeError, TypeError)):
            meta.orientation.align_tango_to_body = (0.0, 0.0, 0.0, 1.0)  # type: ignore[misc]

    def test_errors_is_immutable(
        self, loader: RoninMetadataLoader, valid_recording: Recording
    ) -> None:
        meta = loader.load(valid_recording)
        with pytest.raises((AttributeError, TypeError)):
            meta.errors.ekf_ori_error = 0.0  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# Missing file
# ──────────────────────────────────────────────────────────────────────────────


class TestMissingFile:
    def test_missing_info_json_raises_dataset_error(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path, write_file=False)
        with pytest.raises(DatasetError, match="info.json not found"):
            loader.load(recording)

    def test_error_message_contains_recording_id(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path, write_file=False)
        with pytest.raises(DatasetError, match="a000_1"):
            loader.load(recording)


# ──────────────────────────────────────────────────────────────────────────────
# Malformed JSON
# ──────────────────────────────────────────────────────────────────────────────


class TestMalformedJson:
    def test_invalid_json_raises_dataset_error(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        rec_dir = tmp_path / "a000_1"
        rec_dir.mkdir()
        (rec_dir / "data.hdf5").touch()
        info_path = rec_dir / "info.json"
        info_path.write_text("{ this is not json }", encoding="utf-8")
        recording = Recording(
            recording_id="a000_1",
            split="train_dataset_1",
            recording_path=rec_dir,
            hdf5_path=rec_dir / "data.hdf5",
            info_path=info_path,
        )
        with pytest.raises(DatasetError, match="invalid JSON"):
            loader.load(recording)

    def test_json_array_root_raises_dataset_error(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        """Root-level JSON arrays are not valid info.json schemas."""
        rec_dir = tmp_path / "a000_1"
        rec_dir.mkdir()
        (rec_dir / "data.hdf5").touch()
        info_path = rec_dir / "info.json"
        info_path.write_text("[1, 2, 3]", encoding="utf-8")
        recording = Recording(
            recording_id="a000_1",
            split="train_dataset_1",
            recording_path=rec_dir,
            hdf5_path=rec_dir / "data.hdf5",
            info_path=info_path,
        )
        with pytest.raises(DatasetError, match="JSON object"):
            loader.load(recording)

    def test_empty_file_raises_dataset_error(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        rec_dir = tmp_path / "a000_1"
        rec_dir.mkdir()
        (rec_dir / "data.hdf5").touch()
        info_path = rec_dir / "info.json"
        info_path.write_text("", encoding="utf-8")
        recording = Recording(
            recording_id="a000_1",
            split="train_dataset_1",
            recording_path=rec_dir,
            hdf5_path=rec_dir / "data.hdf5",
            info_path=info_path,
        )
        with pytest.raises(DatasetError, match="invalid JSON"):
            loader.load(recording)


# ──────────────────────────────────────────────────────────────────────────────
# Missing required fields
# ──────────────────────────────────────────────────────────────────────────────


REQUIRED_FIELDS = [
    "length",
    "date",
    "device",
    "type",
    "start_frame",
    "imu_init_gyro_bias",
    "imu_end_gyro_bias",
    "imu_acce_bias",
    "imu_acce_scale",
    "imu_reference_time",
    "tango_reference_time",
    "imu_time_offset",
    "align_tango_to_body",
    "start_calibration",
    "end_calibration",
    "gyro_integration_error",
    "grv_ori_error",
    "ekf_ori_error",
]


class TestMissingRequiredFields:
    @pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
    def test_missing_required_field_raises_dataset_error(
        self,
        loader: RoninMetadataLoader,
        tmp_path: Path,
        missing_field: str,
    ) -> None:
        data = dict(VALID_INFO)
        del data[missing_field]
        recording = _make_recording(tmp_path / missing_field, data)
        with pytest.raises(DatasetError, match=missing_field):
            loader.load(recording)


# ──────────────────────────────────────────────────────────────────────────────
# Type validation
# ──────────────────────────────────────────────────────────────────────────────


class TestTypeValidation:
    def test_wrong_type_for_length_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, length="not_a_number")
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="length"):
            loader.load(recording)

    def test_wrong_type_for_device_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, device=42)
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="device"):
            loader.load(recording)

    def test_wrong_type_for_start_frame_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, start_frame=5269.0)  # float instead of int
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="start_frame"):
            loader.load(recording)

    def test_wrong_length_for_gyro_bias_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, imu_init_gyro_bias=[0.0, 0.0])  # 2 elements instead of 3
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="imu_init_gyro_bias"):
            loader.load(recording)

    def test_wrong_length_for_quaternion_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, align_tango_to_body=[0.5, 0.5, 0.5])  # 3 elements instead of 4
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="align_tango_to_body"):
            loader.load(recording)

    def test_non_numeric_element_in_list_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, imu_acce_scale=[0.9985, "bad", 0.9979])
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="imu_acce_scale"):
            loader.load(recording)

    def test_string_instead_of_list_raises(
        self, loader: RoninMetadataLoader, tmp_path: Path
    ) -> None:
        data = dict(VALID_INFO, imu_acce_bias="not_a_list")
        recording = _make_recording(tmp_path, data)
        with pytest.raises(DatasetError, match="imu_acce_bias"):
            loader.load(recording)
