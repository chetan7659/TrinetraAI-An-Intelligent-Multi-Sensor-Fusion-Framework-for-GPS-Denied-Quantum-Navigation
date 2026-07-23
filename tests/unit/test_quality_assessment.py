"""Unit tests for single-pass quality assessment."""

from __future__ import annotations

from trinetra.analysis.quality_assessment import assess_recording_quality
from trinetra.domain.interfaces.sensor_record import SensorRecord


def test_assess_recording_quality_clean_stream() -> None:
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, 0.0, 1.0),
            gyroscope=(0.1, 0.1, 0.1),
            magnetometer=(10.0, 20.0, 30.0),
            gravity=(0.0, 0.0, 9.8),
            linear_acceleration=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        ),
        SensorRecord(
            frame_id=1,
            timestamp=0.1,
            accelerometer=(0.0, 0.0, 1.1),
            gyroscope=(0.1, 0.2, 0.1),
            magnetometer=(10.0, 21.0, 30.0),
            gravity=(0.0, 0.1, 9.8),
            linear_acceleration=(0.0, 0.0, 0.1),
            orientation=(0.0, 0.1, 0.0, 0.9949874),
        ),
    ]

    res = assess_recording_quality("test_rec", stream)

    assert res["recording_id"] == "test_rec"
    assert res["total_frames"] == 2

    # Temporal consistency
    t_cons = res["temporal_consistency"]
    assert t_cons["dropped_frames"] == 0
    assert t_cons["duplicate_timestamps"] == 0
    assert t_cons["non_monotonic"] == 0

    # Missing samples should be 0 across the board
    assert sum(res["missing_samples"].values()) == 0
    assert sum(res["nan_count"].values()) == 0

    # Scores should be perfect or near perfect
    scores = res["scores"]
    assert scores["completeness_score"] == 1.0
    assert scores["validity_score"] == 1.0
    assert scores["consistency_score"] == 1.0
    assert scores["overall_score"] == 1.0


def test_assess_recording_quality_with_defects() -> None:
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, float("nan"), 1.0),  # NaN
            gyroscope=(0.1, 0.1, 0.1),
            magnetometer=None,  # Missing
            gravity=(0.0, 0.0, 9.8),
            linear_acceleration=(0.0, 0.0, 0.0),
            orientation=(1.0, 1.0, 1.0, 1.0),  # Invalid norm
        ),
        SensorRecord(
            frame_id=1,
            timestamp=0.0,  # Duplicate timestamp
            accelerometer=(160.0, 0.0, 1.1),  # Saturated (Accel > 156.96)
            gyroscope=(0.1, 0.1, 0.1),  # Constant value from previous frame
            magnetometer=None,
            gravity=(0.0, 0.0, 9.8),
            linear_acceleration=(0.0, 0.0, 0.1),
            orientation=(0.0, 0.0, 0.0, 1.0),
        ),
    ]

    res = assess_recording_quality("test_defects", stream)

    assert res["total_frames"] == 2

    assert res["nan_count"]["accelerometer"] == 1
    assert res["missing_samples"]["magnetometer"] == 2
    assert res["invalid_quaternions"] == 1
    assert res["saturation_count"]["accelerometer"] == 1
    assert res["constant_values"]["gyroscope"] == 1

    t_cons = res["temporal_consistency"]
    assert t_cons["duplicate_timestamps"] == 1

    scores = res["scores"]
    assert scores["overall_score"] < 1.0
