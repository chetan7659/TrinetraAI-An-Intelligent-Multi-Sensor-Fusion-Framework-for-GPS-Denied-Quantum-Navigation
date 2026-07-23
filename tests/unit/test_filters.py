"""Unit tests for dataset filtering utilities (M1.6.2)."""

from __future__ import annotations

import pytest

from trinetra.application.dataset.filters import (
    filter_by_frame,
    filter_by_time,
    predicate_filter,
    skip,
    take,
)
from trinetra.domain.interfaces.sensor_record import SensorRecord

# ---------------------------------------------------------------------------
# Synthetic Data Factory
# ---------------------------------------------------------------------------


def _make_record(frame_id: int, timestamp: float) -> SensorRecord:
    """Helper to create minimal SensorRecords for filtering tests."""
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
    timestamps: 0.0..9.0
    """
    return [_make_record(i, float(i)) for i in range(10)]


def test_lazy_evaluation(records: list[SensorRecord]) -> None:
    """Verify that filters return lazy generators, not materialized lists."""
    result = filter_by_frame(records, start=0)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")
    assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# filter_by_frame
# ---------------------------------------------------------------------------


class TestFilterByFrame:
    def test_no_bounds_returns_all(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_frame(records))
        assert len(result) == 10
        assert result == records

    def test_start_bound_inclusive(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_frame(records, start=5))
        assert len(result) == 5
        assert result[0].frame_id == 5
        assert result[-1].frame_id == 9

    def test_end_bound_exclusive(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_frame(records, end=5))
        assert len(result) == 5
        assert result[0].frame_id == 0
        assert result[-1].frame_id == 4

    def test_both_bounds(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_frame(records, start=2, end=6))
        assert len(result) == 4
        assert [r.frame_id for r in result] == [2, 3, 4, 5]

    def test_out_of_bounds_returns_empty(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_frame(records, start=20))
        assert len(result) == 0

    def test_invalid_range_returns_empty(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_frame(records, start=5, end=2))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_by_time
# ---------------------------------------------------------------------------


class TestFilterByTime:
    def test_no_bounds_returns_all(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_time(records))
        assert len(result) == 10

    def test_start_bound_inclusive(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_time(records, start=5.0))
        assert len(result) == 5
        assert result[0].timestamp == 5.0

    def test_end_bound_exclusive(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_time(records, end=5.0))
        assert len(result) == 5
        assert result[-1].timestamp == 4.0

    def test_both_bounds(self, records: list[SensorRecord]) -> None:
        result = list(filter_by_time(records, start=2.5, end=6.5))
        assert len(result) == 4
        assert [r.timestamp for r in result] == [3.0, 4.0, 5.0, 6.0]

    def test_exact_float_matching(self, records: list[SensorRecord]) -> None:
        # frame 3 is 3.0. frame 4 is 4.0.
        result = list(filter_by_time(records, start=3.0, end=4.0))
        assert len(result) == 1
        assert result[0].timestamp == 3.0


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------


class TestTake:
    def test_take_less_than_length(self, records: list[SensorRecord]) -> None:
        result = list(take(records, 3))
        assert len(result) == 3
        assert [r.frame_id for r in result] == [0, 1, 2]

    def test_take_more_than_length(self, records: list[SensorRecord]) -> None:
        result = list(take(records, 20))
        assert len(result) == 10
        assert result == records

    def test_take_zero_returns_empty(self, records: list[SensorRecord]) -> None:
        result = list(take(records, 0))
        assert len(result) == 0

    def test_take_negative_returns_empty(self, records: list[SensorRecord]) -> None:
        result = list(take(records, -5))
        assert len(result) == 0

    def test_take_returns_iterator(self, records: list[SensorRecord]) -> None:
        result = take(records, 5)
        assert hasattr(result, "__iter__")
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# skip
# ---------------------------------------------------------------------------


class TestSkip:
    def test_skip_less_than_length(self, records: list[SensorRecord]) -> None:
        result = list(skip(records, 3))
        assert len(result) == 7
        assert [r.frame_id for r in result] == [3, 4, 5, 6, 7, 8, 9]

    def test_skip_more_than_length(self, records: list[SensorRecord]) -> None:
        result = list(skip(records, 20))
        assert len(result) == 0

    def test_skip_zero_returns_all(self, records: list[SensorRecord]) -> None:
        result = list(skip(records, 0))
        assert len(result) == 10
        assert result == records

    def test_skip_negative_returns_all(self, records: list[SensorRecord]) -> None:
        result = list(skip(records, -5))
        assert len(result) == 10
        assert result == records

    def test_skip_returns_iterator(self, records: list[SensorRecord]) -> None:
        result = skip(records, 5)
        assert hasattr(result, "__iter__")
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# predicate_filter
# ---------------------------------------------------------------------------


class TestPredicateFilter:
    def test_predicate_keeps_matches(self, records: list[SensorRecord]) -> None:
        def is_even(r: SensorRecord) -> bool:
            return r.frame_id % 2 == 0

        result = list(predicate_filter(records, is_even))
        assert len(result) == 5
        assert [r.frame_id for r in result] == [0, 2, 4, 6, 8]

    def test_predicate_drops_all(self, records: list[SensorRecord]) -> None:
        def always_false(r: SensorRecord) -> bool:
            return False

        result = list(predicate_filter(records, always_false))
        assert len(result) == 0

    def test_predicate_keeps_all(self, records: list[SensorRecord]) -> None:
        def always_true(r: SensorRecord) -> bool:
            return True

        result = list(predicate_filter(records, always_true))
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Chaining / Composition
# ---------------------------------------------------------------------------


class TestChaining:
    def test_chaining_filters(self, records: list[SensorRecord]) -> None:
        """Verify that filters can be composed together functionally."""
        # 1. Skip first 2 -> [2..9]
        it1 = skip(records, 2)
        # 2. Take 6 -> [2..7]
        it2 = take(it1, 6)
        # 3. Filter by time start=4.0 -> [4..7]
        it3 = filter_by_time(it2, start=4.0)
        # 4. Filter by frame end=7 -> [4, 5, 6]
        it4 = filter_by_frame(it3, end=7)
        # 5. Predicate (keep evens) -> [4, 6]
        it5 = predicate_filter(it4, lambda r: r.frame_id % 2 == 0)

        result = list(it5)
        assert len(result) == 2
        assert [r.frame_id for r in result] == [4, 6]
