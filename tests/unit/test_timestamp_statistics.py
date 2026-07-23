"""Unit tests for timestamp statistics."""

from __future__ import annotations

from trinetra.analysis.timestamp_statistics import TimestampStatsAggregator
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


def test_timestamp_stats_empty() -> None:
    agg = TimestampStatsAggregator("rec_1", "train")
    result = agg.finalize()
    assert result["recording_id"] == "rec_1"
    assert result["split"] == "train"
    assert result["frame_count"] == 0
    assert result["is_monotonic"] is True
    assert result["duplicate_count"] == 0
    assert result["missing_count"] == 0
    assert result["min_dt"] == 0.0
    assert result["max_dt"] == 0.0
    assert result["avg_dt"] == 0.0


def test_timestamp_stats_single_frame() -> None:
    agg = TimestampStatsAggregator("rec_1", "train")
    agg.update(_make_record(0, 1.0))
    result = agg.finalize()

    assert result["frame_count"] == 1
    assert result["is_monotonic"] is True
    assert result["duplicate_count"] == 0
    assert result["min_dt"] == 0.0
    assert result["max_dt"] == 0.0
    assert result["avg_dt"] == 0.0


def test_timestamp_stats_normal_sequence() -> None:
    agg = TimestampStatsAggregator("rec_1", "train")
    # dt will be 0.5 for all
    agg.update(_make_record(0, 0.0))
    agg.update(_make_record(1, 0.5))
    agg.update(_make_record(2, 1.0))

    result = agg.finalize()

    assert result["frame_count"] == 3
    assert result["is_monotonic"] is True
    assert result["duplicate_count"] == 0
    assert result["min_dt"] == 0.5
    assert result["max_dt"] == 0.5
    assert result["avg_dt"] == 0.5


def test_timestamp_stats_non_monotonic() -> None:
    agg = TimestampStatsAggregator("rec_1", "train")
    agg.update(_make_record(0, 0.0))
    agg.update(_make_record(1, 1.0))
    agg.update(_make_record(2, 0.5))  # Goes backwards

    result = agg.finalize()

    assert result["is_monotonic"] is False
    assert result["duplicate_count"] == 0
    # The valid dt is 1.0, the negative dt is ignored for min/max/avg
    assert result["min_dt"] == 1.0
    assert result["max_dt"] == 1.0


def test_timestamp_stats_duplicates() -> None:
    agg = TimestampStatsAggregator("rec_1", "train")
    agg.update(_make_record(0, 0.0))
    agg.update(_make_record(1, 1.0))
    agg.update(_make_record(2, 1.0))  # Duplicate
    agg.update(_make_record(3, 2.0))

    result = agg.finalize()

    assert result["is_monotonic"] is True
    assert result["duplicate_count"] == 1
    assert result["min_dt"] == 1.0
    assert result["max_dt"] == 1.0
    assert result["avg_dt"] == 1.0
