"""Unit tests for validation summary orchestration and aggregation."""

from __future__ import annotations

from trinetra.analysis.validation_summary import (
    aggregate_validation_results,
    validate_recording_geometry,
)
from trinetra.domain.interfaces.sensor_record import SensorRecord


def test_validate_recording_geometry_clean() -> None:
    # Perfect alignment: A = G + L, perfect unit quats, gravity = 9.81
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, 0.0, 9.81),
            gyroscope=(0.0, 0.0, 0.0),
            magnetometer=(0.0, 0.0, 0.0),
            gravity=(0.0, 0.0, 9.81),
            linear_acceleration=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
    ]
    res = validate_recording_geometry("rec_clean", stream)
    assert res["status"] == "PASS"
    assert res["gravity"]["bias"] == 0.0
    assert res["orientation"]["mean_norm"] == 1.0
    assert res["frame"]["mean_residual"] == 0.0


def test_validate_recording_geometry_review() -> None:
    # Small gravity bias (e.g. 9.95 - 9.81 = 0.14 > 0.1 threshold) -> REVIEW
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, 0.0, 9.95),
            gyroscope=(0.0, 0.0, 0.0),
            magnetometer=(0.0, 0.0, 0.0),
            gravity=(0.0, 0.0, 9.95),
            linear_acceleration=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
    ]
    res = validate_recording_geometry("rec_review", stream)
    assert res["status"] == "REVIEW"


def test_validate_recording_geometry_fail() -> None:
    # Huge gravity bias (e.g. 11.5 - 9.81 = 1.69 > 1.0 threshold) -> FAIL
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, 0.0, 11.5),
            gyroscope=(0.0, 0.0, 0.0),
            magnetometer=(0.0, 0.0, 0.0),
            gravity=(0.0, 0.0, 11.5),
            linear_acceleration=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
    ]
    res = validate_recording_geometry("rec_fail", stream)
    assert res["status"] == "FAIL"


def test_validate_recording_geometry_missing_optional() -> None:
    # No gravity or orientation
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, 0.0, 10.0),
            gyroscope=None,
            magnetometer=None,
            gravity=None,
            linear_acceleration=None,
            orientation=None,
        )
    ]
    res = validate_recording_geometry("rec_missing", stream)
    assert res["status"] == "PASS"  # Passes since the properties we check are empty or valid
    assert res["gravity"]["mean_magnitude"] == 0.0
    assert res["orientation"]["mean_norm"] == 0.0
    assert res["frame"]["mean_residual"] == 0.0


def test_validate_recording_geometry_empty() -> None:
    res = validate_recording_geometry("rec_empty", [])
    assert res["status"] == "PASS"
    assert res["gravity"]["mean_magnitude"] == 0.0


def test_aggregate_validation_results() -> None:
    results = [
        {
            "recording_id": "rec_01",
            "status": "PASS",
            "gravity": {"mean_magnitude": 9.81, "bias": 0.0},
            "coordinate": {
                "accel_mean_magnitude": 9.81,
                "gravity_alignment_axis": 2,
                "axis_polarities_consistent": 1.0,
            },
            "orientation": {
                "min_norm": 1.0,
                "max_norm": 1.0,
                "mean_norm": 1.0,
                "sign_flip_count": 0.0,
                "abnormal_jump_count": 0.0,
                "mean_angular_step_rad": 0.0,
                "max_angular_step_rad": 0.0,
            },
            "frame": {
                "mean_residual": 0.0,
                "rms_residual": 0.0,
                "max_residual": 0.0,
                "p95_residual": 0.0,
            },
        }
    ]
    dfs = aggregate_validation_results(results)
    assert len(dfs) == 3
    assert dfs["coordinate_checks"].iloc[0]["Recording"] == "rec_01"
    assert dfs["coordinate_checks"].iloc[0]["Status"] == "PASS"
    assert dfs["frame_validation"].iloc[0]["RMS Residual"] == 0.0
