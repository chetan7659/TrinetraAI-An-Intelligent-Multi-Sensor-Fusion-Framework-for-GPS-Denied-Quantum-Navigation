"""Unit tests for SplitLoader (M1.6.4).

Collaborators (Adapter, RecordingIterator) are replaced with lightweight stubs.
No real dataset, HDF5 file, or filesystem access is required.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Iterator
from typing import Any

import pytest

from trinetra.application.dataset.split_loader import SplitLoader
from trinetra.domain.interfaces.sensor_record import SensorRecord
from trinetra.shared.exceptions import DatasetError

# ---------------------------------------------------------------------------
# Synthetic Data Factory
# ---------------------------------------------------------------------------

# Opaque sentinel objects representing individual recordings
REC_1 = object()
REC_2 = object()
REC_3 = object()


def _make_record(frame_id: int) -> SensorRecord:
    return SensorRecord(
        frame_id=frame_id,
        timestamp=float(frame_id),
        accelerometer=(0.0, 0.0, 0.0),
        gyroscope=(0.0, 0.0, 0.0),
        linear_acceleration=(0.0, 0.0, 0.0),
        gravity=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        magnetometer=(0.0, 0.0, 0.0),
    )


# Three records per recording to verify chaining order
RECORDS_FOR_REC_1 = [_make_record(0), _make_record(1), _make_record(2)]
RECORDS_FOR_REC_2 = [_make_record(3), _make_record(4), _make_record(5)]
RECORDS_FOR_REC_3 = [_make_record(6), _make_record(7), _make_record(8)]

RECORDING_MAP = {
    REC_1: RECORDS_FOR_REC_1,
    REC_2: RECORDS_FOR_REC_2,
    REC_3: RECORDS_FOR_REC_3,
}

# ---------------------------------------------------------------------------
# Collaborator stubs
# ---------------------------------------------------------------------------


class StubAdapter:
    """Returns fixed lists of recordings for requested splits."""

    SPLITS: typing.ClassVar[dict[str, list[Any]]] = {
        "train": [REC_1, REC_2],
        "seen": [REC_3],
        "empty": [],
    }

    def __init__(self) -> None:
        self.call_count = 0
        self.received_split: str | None = None

    def list_recordings(self, split: str) -> list[Any]:
        self.call_count += 1
        self.received_split = split
        if split not in self.SPLITS:
            raise DatasetError(f"Split not found: {split}")
        return self.SPLITS[split]


class StubRecordingIterator:
    """Yields records corresponding to the requested recording sentinel."""

    def __init__(self) -> None:
        self.call_count = 0
        self.received_recordings: list[Any] = []

    def iter_recording(self, recording: Any) -> Iterator[SensorRecord]:
        self.call_count += 1
        self.received_recordings.append(recording)
        yield from RECORDING_MAP[recording]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_adapter() -> StubAdapter:
    return StubAdapter()


@pytest.fixture()
def stub_iterator() -> StubRecordingIterator:
    return StubRecordingIterator()


@pytest.fixture()
def loader(
    stub_adapter: StubAdapter,
    stub_iterator: StubRecordingIterator,
) -> SplitLoader:
    return SplitLoader(stub_adapter, stub_iterator)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_accepts_injected_collaborators(
        self,
        stub_adapter: StubAdapter,
        stub_iterator: StubRecordingIterator,
    ) -> None:
        it = SplitLoader(stub_adapter, stub_iterator)
        assert it is not None

    def test_stores_collaborators_as_private_attributes(
        self,
        stub_adapter: StubAdapter,
        stub_iterator: StubRecordingIterator,
    ) -> None:
        it = SplitLoader(stub_adapter, stub_iterator)
        assert it._adapter is stub_adapter
        assert it._recording_iterator is stub_iterator


# ---------------------------------------------------------------------------
# iter_recordings (Convenience Wrapper)
# ---------------------------------------------------------------------------


class TestIterRecordings:
    def test_delegates_to_adapter(self, loader: SplitLoader, stub_adapter: StubAdapter) -> None:
        result = list(loader.iter_recordings("train"))
        assert stub_adapter.call_count == 1
        assert stub_adapter.received_split == "train"
        assert result == [REC_1, REC_2]

    def test_laziness(self, loader: SplitLoader) -> None:
        result = loader.iter_recordings("train")
        assert hasattr(result, "__iter__")
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# iter_split (Main Orchestration)
# ---------------------------------------------------------------------------


class TestIterSplit:
    def test_adapter_called_exactly_once(
        self, loader: SplitLoader, stub_adapter: StubAdapter
    ) -> None:
        list(loader.iter_split("train"))
        assert stub_adapter.call_count == 1
        assert stub_adapter.received_split == "train"

    def test_iterator_called_for_every_recording(
        self, loader: SplitLoader, stub_iterator: StubRecordingIterator
    ) -> None:
        list(loader.iter_split("train"))
        # 'train' has REC_1 and REC_2
        assert stub_iterator.call_count == 2
        assert stub_iterator.received_recordings == [REC_1, REC_2]

    def test_iterator_not_called_if_split_empty(
        self, loader: SplitLoader, stub_iterator: StubRecordingIterator
    ) -> None:
        result = list(loader.iter_split("empty"))
        assert stub_iterator.call_count == 0
        assert len(result) == 0

    def test_yields_all_records_in_order(self, loader: SplitLoader) -> None:
        result = list(loader.iter_split("train"))
        assert len(result) == 6
        expected_frames = [0, 1, 2, 3, 4, 5]
        actual_frames = [r.frame_id for r in result]
        assert actual_frames == expected_frames

    def test_yields_single_recording_correctly(self, loader: SplitLoader) -> None:
        result = list(loader.iter_split("seen"))
        assert len(result) == 3
        expected_frames = [6, 7, 8]
        actual_frames = [r.frame_id for r in result]
        assert actual_frames == expected_frames

    def test_returns_generator(self, loader: SplitLoader) -> None:
        result = loader.iter_split("train")
        assert isinstance(result, types.GeneratorType)


# ---------------------------------------------------------------------------
# Exception Propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_adapter_exception_propagates(self, loader: SplitLoader) -> None:
        with pytest.raises(DatasetError, match="Split not found: invalid"):
            list(loader.iter_split("invalid"))

    def test_iterator_exception_propagates(self, stub_adapter: StubAdapter) -> None:
        class StubIteratorRaises:
            def iter_recording(self, recording: Any) -> Iterator[SensorRecord]:
                raise DatasetError("simulated iteration failure")
                yield

        loader = SplitLoader(stub_adapter, StubIteratorRaises())

        with pytest.raises(DatasetError, match="simulated iteration failure"):
            list(loader.iter_split("train"))
