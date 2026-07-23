"""RoNIN info.json metadata loader.

This module provides :class:`RoninMetadataLoader`, which reads and validates
the ``info.json`` metadata file for a single recording and converts it into an
immutable :class:`~.metadata_models.RoninRecordingMetadata` value object.

Responsibilities
----------------
* Open ``recording.info_path`` (the only file I/O this module performs).
* Parse the JSON content.
* Validate required fields are present and have the correct types.
* Return an immutable :class:`~.metadata_models.RoninRecordingMetadata`.

Non-responsibilities
--------------------
* Opening or inspecting ``data.hdf5``     → future :mod:`hdf5_reader` component
* Preprocessing or resampling sensor data → future preprocessing pipeline
* Computing statistics from sensor arrays  → future EDA pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trinetra.shared.exceptions import DatasetError

from .metadata_models import (
    ImuCalibration,
    OrientationCalibration,
    OrientationErrors,
    RoninRecordingMetadata,
    TimeSynchronization,
)
from .models import Recording

# ──────────────────────────────────────────────────────────────────────────────
# Field-level validation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _require_float(data: dict[str, Any], key: str, ctx: str) -> float:
    """Extract a required numeric (float/int) field, raising DatasetError if absent or wrong type."""
    if key not in data:
        raise DatasetError(f"[{ctx}] Required field '{key}' is missing from info.json.")
    value = data[key]
    if not isinstance(value, int | float):
        raise DatasetError(
            f"[{ctx}] Field '{key}' must be a number, got {type(value).__name__}: {value!r}."
        )
    return float(value)


def _require_int(data: dict[str, Any], key: str, ctx: str) -> int:
    """Extract a required integer field, raising DatasetError if absent or wrong type."""
    if key not in data:
        raise DatasetError(f"[{ctx}] Required field '{key}' is missing from info.json.")
    value = data[key]
    if not isinstance(value, int):
        raise DatasetError(
            f"[{ctx}] Field '{key}' must be an integer, got {type(value).__name__}: {value!r}."
        )
    return value


def _require_str(data: dict[str, Any], key: str, ctx: str) -> str:
    """Extract a required string field, raising DatasetError if absent or wrong type."""
    if key not in data:
        raise DatasetError(f"[{ctx}] Required field '{key}' is missing from info.json.")
    value = data[key]
    if not isinstance(value, str):
        raise DatasetError(
            f"[{ctx}] Field '{key}' must be a string, got {type(value).__name__}: {value!r}."
        )
    return value


def _require_float_list(
    data: dict[str, Any], key: str, expected_len: int, ctx: str
) -> tuple[float, ...]:
    """Extract a required list-of-numbers field with a fixed expected length."""
    if key not in data:
        raise DatasetError(f"[{ctx}] Required field '{key}' is missing from info.json.")
    value = data[key]
    if not isinstance(value, list):
        raise DatasetError(
            f"[{ctx}] Field '{key}' must be a list, got {type(value).__name__}: {value!r}."
        )
    if len(value) != expected_len:
        raise DatasetError(
            f"[{ctx}] Field '{key}' must have exactly {expected_len} elements, "
            f"got {len(value)}: {value!r}."
        )
    for i, elem in enumerate(value):
        if not isinstance(elem, int | float):
            raise DatasetError(
                f"[{ctx}] Field '{key}[{i}]' must be a number, "
                f"got {type(elem).__name__}: {elem!r}."
            )
    return tuple(float(v) for v in value)


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────


class RoninMetadataLoader:
    """Loads and validates ``info.json`` for a single RoNIN recording.

    Usage::

        adapter  = RoninAdapter()
        recording = adapter.get_recording("a000_1")

        loader   = RoninMetadataLoader()
        metadata = loader.load(recording)

        print(metadata.device)          # e.g. "asus5"
        print(metadata.length)          # e.g. 338.04 (seconds)
        print(metadata.calibration.imu_acce_scale)

    Raises:
        DatasetError: On any I/O, parsing, or validation failure.

    Notes:
        This class is stateless; a single instance can be reused to load
        metadata for multiple recordings without any side-effects.
    """

    def load(self, recording: Recording) -> RoninRecordingMetadata:
        """Parse and validate ``info.json`` for the given recording.

        Args:
            recording: A :class:`~.models.Recording` instance returned by
                :class:`~.adapter.RoninAdapter`.

        Returns:
            An immutable :class:`~.metadata_models.RoninRecordingMetadata`
            representing the full contents of ``info.json``.

        Raises:
            DatasetError: If:

                * ``recording.info_path`` does not exist,
                * the file contains invalid JSON,
                * a required field is absent,
                * a field contains an unexpected type.
        """
        raw = self._read_json(recording.info_path, recording.recording_id)
        return self._parse(raw, recording)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_json(info_path: Path, recording_id: str) -> dict[str, Any]:
        """Read and JSON-parse the info file, raising DatasetError on failure."""
        if not info_path.exists():
            raise DatasetError(f"[{recording_id}] info.json not found at: {info_path}")
        if not info_path.is_file():
            raise DatasetError(f"[{recording_id}] info.json path is not a file: {info_path}")
        try:
            text = info_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetError(f"[{recording_id}] Failed to read info.json: {exc}") from exc

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"[{recording_id}] info.json contains invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise DatasetError(
                f"[{recording_id}] info.json root must be a JSON object, "
                f"got {type(data).__name__}."
            )
        return data

    @staticmethod
    def _parse(data: dict[str, Any], recording: Recording) -> RoninRecordingMetadata:
        """Map a raw JSON dict to a validated :class:`RoninRecordingMetadata`."""
        ctx = recording.recording_id

        # ── Sequence-level fields ─────────────────────────────────────────────
        length = _require_float(data, "length", ctx)
        date = _require_str(data, "date", ctx)
        device = _require_str(data, "device", ctx)
        recording_type = _require_str(data, "type", ctx)
        start_frame = _require_int(data, "start_frame", ctx)

        # ── IMU Calibration ───────────────────────────────────────────────────
        calibration = ImuCalibration(
            imu_init_gyro_bias=_require_float_list(data, "imu_init_gyro_bias", 3, ctx),
            imu_end_gyro_bias=_require_float_list(data, "imu_end_gyro_bias", 3, ctx),
            imu_acce_bias=_require_float_list(data, "imu_acce_bias", 3, ctx),
            imu_acce_scale=_require_float_list(data, "imu_acce_scale", 3, ctx),
        )

        # ── Time Synchronisation ──────────────────────────────────────────────
        time_sync = TimeSynchronization(
            imu_reference_time=_require_float(data, "imu_reference_time", ctx),
            tango_reference_time=_require_float(data, "tango_reference_time", ctx),
            imu_time_offset=_require_float(data, "imu_time_offset", ctx),
        )

        # ── Orientation Calibration ───────────────────────────────────────────
        orientation = OrientationCalibration(
            align_tango_to_body=_require_float_list(data, "align_tango_to_body", 4, ctx),
            start_calibration=_require_float_list(data, "start_calibration", 4, ctx),
            end_calibration=_require_float_list(data, "end_calibration", 4, ctx),
        )

        # ── Orientation Errors ────────────────────────────────────────────────
        errors = OrientationErrors(
            gyro_integration_error=_require_float(data, "gyro_integration_error", ctx),
            grv_ori_error=_require_float(data, "grv_ori_error", ctx),
            ekf_ori_error=_require_float(data, "ekf_ori_error", ctx),
        )

        return RoninRecordingMetadata(
            recording_id=recording.recording_id,
            split=recording.split,
            length=length,
            date=date,
            device=device,
            recording_type=recording_type,
            start_frame=start_frame,
            calibration=calibration,
            time_sync=time_sync,
            orientation=orientation,
            errors=errors,
        )
