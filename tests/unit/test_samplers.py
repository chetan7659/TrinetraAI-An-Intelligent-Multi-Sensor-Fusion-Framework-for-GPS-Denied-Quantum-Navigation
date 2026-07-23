"""Unit tests for dataset sampling and batching utilities (M1.6.3)."""

from __future__ import annotations

import pytest

from trinetra.application.dataset.filters import take
from trinetra.application.dataset.samplers import batch, chunk_by_time, stride, window
from trinetra.domain.interfaces.sensor_record import SensorRecord

# ---------------------------------------------------------------------------
# Synthetic Data Factory
# ---------------------------------------------------------------------------


def _make_record(frame_id: int, timestamp: float) -> SensorRecord:
    """Helper to create minimal SensorRecords for sampling tests."""
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


@pytest.fixture()
def records() -> list[SensorRecord]:
    """Return a stream of 10 sequential records.

    frame_ids: 0..9
    timestamps: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
    """
    return [_make_record(i, float(i) * 0.5) for i in range(10)]


# ---------------------------------------------------------------------------
# batch()
# ---------------------------------------------------------------------------


class TestBatch:
    def test_even_batches(self, records: list[SensorRecord]) -> None:
        result = list(batch(records, batch_size=2))
        assert len(result) == 5
        assert [len(b) for b in result] == [2, 2, 2, 2, 2]
        assert [r.frame_id for r in result[0]] == [0, 1]
        assert [r.frame_id for r in result[4]] == [8, 9]

    def test_partial_final_batch(self, records: list[SensorRecord]) -> None:
        result = list(batch(records, batch_size=3))
        assert len(result) == 4
        assert [len(b) for b in result] == [3, 3, 3, 1]
        assert [r.frame_id for r in result[3]] == [9]

    def test_batch_larger_than_input(self, records: list[SensorRecord]) -> None:
        result = list(batch(records, batch_size=20))
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_invalid_batch_size(self, records: list[SensorRecord]) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            list(batch(records, batch_size=0))
        with pytest.raises(ValueError, match="must be > 0"):
            list(batch(records, batch_size=-1))

    def test_empty_input(self) -> None:
        result = list(batch([], batch_size=3))
        assert len(result) == 0

    def test_laziness(self, records: list[SensorRecord]) -> None:
        result = batch(records, batch_size=2)
        assert hasattr(result, "__iter__")
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# window()
# ---------------------------------------------------------------------------


class TestWindow:
    def test_sliding_window(self, records: list[SensorRecord]) -> None:
        result = list(window(records, size=3))
        # N=10, size=3 -> 8 windows (indices: 012, 123, 234, 345, 456, 567, 678, 789)
        assert len(result) == 8
        assert tuple(r.frame_id for r in result[0]) == (0, 1, 2)
        assert tuple(r.frame_id for r in result[1]) == (1, 2, 3)
        assert tuple(r.frame_id for r in result[-1]) == (7, 8, 9)

    def test_window_size_equals_length(self, records: list[SensorRecord]) -> None:
        result = list(window(records, size=10))
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_window_larger_than_input(self, records: list[SensorRecord]) -> None:
        result = list(window(records, size=11))
        assert len(result) == 0

    def test_invalid_window_size(self, records: list[SensorRecord]) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            list(window(records, size=0))

    def test_empty_input(self) -> None:
        result = list(window([], size=3))
        assert len(result) == 0

    def test_laziness(self, records: list[SensorRecord]) -> None:
        result = window(records, size=3)
        assert hasattr(result, "__iter__")
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# stride()
# ---------------------------------------------------------------------------


class TestStride:
    def test_stride_step_two(self, records: list[SensorRecord]) -> None:
        result = list(stride(records, step=2))
        assert len(result) == 5
        assert [r.frame_id for r in result] == [0, 2, 4, 6, 8]

    def test_stride_step_three(self, records: list[SensorRecord]) -> None:
        result = list(stride(records, step=3))
        assert len(result) == 4
        assert [r.frame_id for r in result] == [0, 3, 6, 9]

    def test_stride_step_larger_than_input(self, records: list[SensorRecord]) -> None:
        result = list(stride(records, step=20))
        assert len(result) == 1
        assert result[0].frame_id == 0

    def test_invalid_step(self, records: list[SensorRecord]) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            list(stride(records, step=0))

    def test_empty_input(self) -> None:
        result = list(stride([], step=2))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# chunk_by_time()
# ---------------------------------------------------------------------------


class TestChunkByTime:
    def test_exact_time_boundaries(self, records: list[SensorRecord]) -> None:
        # Timestamps: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
        # chunk duration: 1.0
        # chunk 1: [0.0, 0.5]
        # chunk 2: [1.0, 1.5]
        # chunk 3: [2.0, 2.5]
        # chunk 4: [3.0, 3.5]
        # chunk 5: [4.0, 4.5]
        result = list(chunk_by_time(records, duration=1.0))
        assert len(result) == 5
        assert [len(c) for c in result] == [2, 2, 2, 2, 2]
        assert [r.timestamp for r in result[0]] == [0.0, 0.5]
        assert [r.timestamp for r in result[1]] == [1.0, 1.5]

    def test_partial_final_time_chunk(self, records: list[SensorRecord]) -> None:
        # duration: 2.0
        # chunk 1: [0.0, 0.5, 1.0, 1.5]  (len 4)
        # chunk 2: [2.0, 2.5, 3.0, 3.5]  (len 4)
        # chunk 3: [4.0, 4.5]            (len 2)
        result = list(chunk_by_time(records, duration=2.0))
        assert len(result) == 3
        assert [len(c) for c in result] == [4, 4, 2]

    def test_duration_larger_than_total_time(self, records: list[SensorRecord]) -> None:
        result = list(chunk_by_time(records, duration=10.0))
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_invalid_duration(self, records: list[SensorRecord]) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            list(chunk_by_time(records, duration=0.0))
        with pytest.raises(ValueError, match="must be > 0"):
            list(chunk_by_time(records, duration=-1.0))

    def test_empty_input(self) -> None:
        result = list(chunk_by_time([], duration=1.0))
        assert len(result) == 0

    def test_gaps_in_time(self) -> None:
        """Verify boundaries are anchored to the first frame even if there are gaps."""
        records = [
            _make_record(0, 0.0),
            _make_record(1, 0.5),
            # Gap of 3 seconds...
            _make_record(2, 3.5),
            _make_record(3, 4.0),
        ]
        # duration 2.0.
        # chunk 1: [0.0, 2.0) -> gets 0.0, 0.5
        # chunk 2: [2.0, 4.0) -> gets 3.5
        # chunk 3: [4.0, 6.0) -> gets 4.0
        result = list(chunk_by_time(records, duration=2.0))
        assert len(result) == 3
        assert [r.timestamp for r in result[0]] == [0.0, 0.5]
        assert [r.timestamp for r in result[1]] == [3.5]
        assert [r.timestamp for r in result[2]] == [4.0]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_compose_filters_and_samplers(self, records: list[SensorRecord]) -> None:
        # Take first 8
        it1 = take(records, 8)
        # Stride by 2 -> 0, 2, 4, 6 (timestamps 0.0, 1.0, 2.0, 3.0)
        it2 = stride(it1, step=2)
        # Batch by 2 -> [[0, 2], [4, 6]]
        result = list(batch(it2, batch_size=2))

        assert len(result) == 2
        assert [r.frame_id for r in result[0]] == [0, 2]
        assert [r.frame_id for r in result[1]] == [4, 6]
