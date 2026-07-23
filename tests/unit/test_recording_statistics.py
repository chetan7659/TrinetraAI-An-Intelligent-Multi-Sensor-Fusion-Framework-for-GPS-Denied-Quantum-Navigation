"""Unit tests for recording statistics."""

from __future__ import annotations

from trinetra.analysis.recording_statistics import RecordingStatsAggregator
from trinetra.domain.interfaces.sensor_record import SensorRecord


def _make_record(frame_id: int, timestamp: float) -> SensorRecord:
    return SensorRecord(
        frame_id=frame_id,
        timestamp=timestamp,
        accelerometer=(0.0, 0.0, 0.0),
        gyroscope=(0.0, 0.0, 0.0),
        linear_acceleration=(0.0, 0.0, 0.0),
        gravity=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        magnetometer=(0.0, 0.0, 0.0),
    )


def test_recording_stats_empty() -> None:
    agg = RecordingStatsAggregator("rec_1", "train")
    result = agg.finalize()
    assert result["recording_id"] == "rec_1"
    assert result["split"] == "train"
    assert result["frame_count"] == 0
    assert result["duration"] == 0.0
    assert result["sampling_frequency"] == 0.0
    assert result["start_timestamp"] is None
    assert result["end_timestamp"] is None


def test_recording_stats_single_frame() -> None:
    agg = RecordingStatsAggregator("rec_1", "train")
    agg.update(_make_record(0, 1.0))
    result = agg.finalize()

    assert result["frame_count"] == 1
    assert result["duration"] == 0.0
    assert result["sampling_frequency"] == 0.0
    assert result["start_timestamp"] == 1.0
    assert result["end_timestamp"] == 1.0


def test_recording_stats_multiple_frames() -> None:
    agg = RecordingStatsAggregator("rec_1", "train")
    # 5 frames, over 2.0 seconds (0.0 to 2.0).
    # Sampling frequency = 5 / 2.0 = 2.5 Hz
    agg.update(_make_record(0, 0.0))
    agg.update(_make_record(1, 0.5))
    agg.update(_make_record(2, 1.0))
    agg.update(_make_record(3, 1.5))
    agg.update(_make_record(4, 2.0))

    result = agg.finalize()

    assert result["frame_count"] == 5
    assert result["duration"] == 2.0
    assert result["sampling_frequency"] == 2.5
    assert result["start_timestamp"] == 0.0
    assert result["end_timestamp"] == 2.0
