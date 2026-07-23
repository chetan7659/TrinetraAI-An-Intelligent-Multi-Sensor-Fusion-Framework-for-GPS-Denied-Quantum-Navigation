"""Unit tests for RecordingIterator (M1.6.1).

All collaborators (MetadataLoader, HDF5Reader, CanonicalMapper) are
replaced with lightweight stubs defined in this file.  No real dataset,
HDF5 file, or filesystem access is required.

Stub design
-----------
Each stub is a minimal class that satisfies the collaborator protocol and
records how it was called so tests can assert call counts and order.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

import pytest

from trinetra.application.dataset.recording_iterator import RecordingIterator
from trinetra.domain.interfaces.sensor_record import SensorRecord
from trinetra.shared.exceptions import DatasetError

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

# A minimal sentinel "recording" object — the iterator passes it through
# without inspecting it, so the type is irrelevant for these tests.
FAKE_RECORDING = object()

# Synthetic SensorRecord instances (fully typed, no HDF5 needed).
_RECORD_A = SensorRecord(
    frame_id=0,
    timestamp=1.0,
    accelerometer=(0.1, 0.2, 9.8),
    gyroscope=(0.01, 0.02, 0.03),
    linear_acceleration=(0.1, 0.2, 0.0),
    gravity=(0.0, 0.0, 9.81),
    orientation=(0.0, 0.0, 0.0, 1.0),
    magnetometer=(10.0, 20.0, 30.0),
)
_RECORD_B = SensorRecord(
    frame_id=1,
    timestamp=1.005,
    accelerometer=(0.11, 0.21, 9.81),
    gyroscope=(0.02, 0.03, 0.04),
    linear_acceleration=(0.11, 0.21, 0.01),
    gravity=(0.0, 0.0, 9.81),
    orientation=(0.01, 0.02, 0.03, 0.99),
    magnetometer=(11.0, 21.0, 31.0),
)
_RECORD_C = SensorRecord(
    frame_id=2,
    timestamp=1.010,
    accelerometer=(0.12, 0.22, 9.82),
    gyroscope=(0.03, 0.04, 0.05),
    linear_acceleration=(0.12, 0.22, 0.02),
    gravity=(0.0, 0.0, 9.81),
    orientation=(0.02, 0.03, 0.04, 0.98),
    magnetometer=(12.0, 22.0, 32.0),
)

ALL_RECORDS = [_RECORD_A, _RECORD_B, _RECORD_C]


# ---------------------------------------------------------------------------
# Collaborator stubs
# ---------------------------------------------------------------------------


class StubMetadataLoader:
    """Returns a fixed sentinel metadata object; records call count."""

    METADATA = object()  # opaque sentinel

    def __init__(self) -> None:
        self.call_count = 0
        self.received_recording: Any = None

    def load(self, recording: Any) -> Any:
        self.call_count += 1
        self.received_recording = recording
        return self.METADATA


class StubHDF5Reader:
    """Returns a fixed list of sentinel raw frames; records call args."""

    RAW_FRAMES: ClassVar[list[object]] = [object(), object(), object()]

    def __init__(self) -> None:
        self.call_count = 0
        self.received_recording: Any = None
        self.received_metadata: Any = None

    def read(self, recording: Any, metadata: Any) -> Iterator[Any]:
        self.call_count += 1
        self.received_recording = recording
        self.received_metadata = metadata
        yield from self.RAW_FRAMES


class StubCanonicalMapper:
    """Maps any iterable to ALL_RECORDS; records call count."""

    def __init__(self) -> None:
        self.call_count = 0
        self.received_frames: list[Any] = []

    def map_frames(self, frames: Iterable[Any]) -> Iterator[SensorRecord]:
        self.call_count += 1
        # Consume and record the raw frames so tests can inspect them.
        self.received_frames = list(frames)
        yield from ALL_RECORDS


class StubMetadataLoaderRaises:
    """Always raises DatasetError from load()."""

    def load(self, recording: Any) -> Any:
        raise DatasetError("[StubMetadataLoaderRaises] Simulated load failure.")


class StubHDF5ReaderRaises:
    """Returns metadata but raises DatasetError from read()."""

    METADATA = object()

    def load(self, recording: Any) -> Any:
        return self.METADATA

    def read(self, recording: Any, metadata: Any) -> Iterator[Any]:
        raise DatasetError("[StubHDF5ReaderRaises] Simulated read failure.")
        yield  # make it a generator (unreachable)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_loader() -> StubMetadataLoader:
    return StubMetadataLoader()


@pytest.fixture()
def stub_reader() -> StubHDF5Reader:
    return StubHDF5Reader()


@pytest.fixture()
def stub_mapper() -> StubCanonicalMapper:
    return StubCanonicalMapper()


@pytest.fixture()
def iterator(
    stub_loader: StubMetadataLoader,
    stub_reader: StubHDF5Reader,
    stub_mapper: StubCanonicalMapper,
) -> RecordingIterator:
    return RecordingIterator(
        metadata_loader=stub_loader,
        hdf5_reader=stub_reader,
        canonical_mapper=stub_mapper,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_accepts_injected_collaborators(
        self,
        stub_loader: StubMetadataLoader,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        it = RecordingIterator(stub_loader, stub_reader, stub_mapper)
        assert it is not None

    def test_stores_collaborators_as_private_attributes(
        self,
        stub_loader: StubMetadataLoader,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        it = RecordingIterator(stub_loader, stub_reader, stub_mapper)
        assert it._metadata_loader is stub_loader
        assert it._hdf5_reader is stub_reader
        assert it._canonical_mapper is stub_mapper


# ---------------------------------------------------------------------------
# Collaborator call counts and arguments
# ---------------------------------------------------------------------------


class TestCollaboratorCalls:
    def test_metadata_loader_called_exactly_once(
        self,
        iterator: RecordingIterator,
        stub_loader: StubMetadataLoader,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_loader.call_count == 1

    def test_metadata_loader_receives_recording(
        self,
        iterator: RecordingIterator,
        stub_loader: StubMetadataLoader,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_loader.received_recording is FAKE_RECORDING

    def test_hdf5_reader_called_exactly_once(
        self,
        iterator: RecordingIterator,
        stub_reader: StubHDF5Reader,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_reader.call_count == 1

    def test_hdf5_reader_receives_recording(
        self,
        iterator: RecordingIterator,
        stub_reader: StubHDF5Reader,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_reader.received_recording is FAKE_RECORDING

    def test_hdf5_reader_receives_metadata_from_loader(
        self,
        iterator: RecordingIterator,
        stub_reader: StubHDF5Reader,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_reader.received_metadata is StubMetadataLoader.METADATA

    def test_canonical_mapper_called_exactly_once(
        self,
        iterator: RecordingIterator,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_mapper.call_count == 1

    def test_canonical_mapper_receives_raw_frames_from_reader(
        self,
        iterator: RecordingIterator,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        # The mapper stub materialises the frames; they must equal the reader's output.
        assert stub_mapper.received_frames == StubHDF5Reader.RAW_FRAMES

    def test_call_order_loader_then_reader_then_mapper(
        self,
        stub_loader: StubMetadataLoader,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        """Verify that load() is called before read() and read() before map_frames().

        Each stub logs at method-call time (not inside a lazy generator body)
        so the order is recorded immediately when the method is invoked.
        """
        call_order: list[str] = []

        class OrderedLoader:
            def load(self, recording: Any) -> Any:
                call_order.append("loader.load")
                return object()

        class OrderedReader:
            def read(self, recording: Any, metadata: Any) -> Iterator[Any]:
                # Log immediately at call time — return a regular iterator,
                # NOT a generator function, so the log fires unconditionally.
                call_order.append("reader.read")
                return iter([])

        class OrderedMapper:
            def map_frames(self, frames: Iterable[Any]) -> Iterator[SensorRecord]:
                call_order.append("mapper.map_frames")
                return iter([])

        it = RecordingIterator(OrderedLoader(), OrderedReader(), OrderedMapper())
        list(it.iter_recording(FAKE_RECORDING))

        assert call_order.index("loader.load") < call_order.index("reader.read")
        assert call_order.index("reader.read") < call_order.index("mapper.map_frames")


# ---------------------------------------------------------------------------
# Output correctness
# ---------------------------------------------------------------------------


class TestOutput:
    def test_yields_sensor_records(self, iterator: RecordingIterator) -> None:
        records = list(iterator.iter_recording(FAKE_RECORDING))
        assert all(isinstance(r, SensorRecord) for r in records)

    def test_yields_correct_number_of_records(self, iterator: RecordingIterator) -> None:
        records = list(iterator.iter_recording(FAKE_RECORDING))
        assert len(records) == len(ALL_RECORDS)

    def test_yields_records_in_order(self, iterator: RecordingIterator) -> None:
        records = list(iterator.iter_recording(FAKE_RECORDING))
        assert records[0] == _RECORD_A
        assert records[1] == _RECORD_B
        assert records[2] == _RECORD_C


# ---------------------------------------------------------------------------
# Generator / laziness
# ---------------------------------------------------------------------------


class TestLaziness:
    def test_iter_recording_returns_iterator(self, iterator: RecordingIterator) -> None:
        result = iterator.iter_recording(FAKE_RECORDING)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_iter_recording_returns_generator(self, iterator: RecordingIterator) -> None:
        result = iterator.iter_recording(FAKE_RECORDING)
        assert isinstance(result, types.GeneratorType)

    def test_metadata_loaded_before_first_frame_consumed(
        self,
        stub_loader: StubMetadataLoader,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        """
        Metadata loading is eager (called before iteration starts).
        This is required because the HDF5 reader needs metadata (start_frame)
        before it can open the file correctly.
        """
        it = RecordingIterator(stub_loader, stub_reader, stub_mapper)
        gen = it.iter_recording(FAKE_RECORDING)
        # Advance to the first frame.
        next(gen)
        # Loader must have been called.
        assert stub_loader.call_count == 1

    def test_iterating_twice_calls_collaborators_twice(
        self,
        stub_loader: StubMetadataLoader,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
        iterator: RecordingIterator,
    ) -> None:
        list(iterator.iter_recording(FAKE_RECORDING))
        list(iterator.iter_recording(FAKE_RECORDING))
        assert stub_loader.call_count == 2
        assert stub_reader.call_count == 2
        assert stub_mapper.call_count == 2


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_metadata_loader_exception_propagates(
        self,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        it = RecordingIterator(
            metadata_loader=StubMetadataLoaderRaises(),
            hdf5_reader=stub_reader,
            canonical_mapper=stub_mapper,
        )
        with pytest.raises(DatasetError, match="load failure"):
            list(it.iter_recording(FAKE_RECORDING))

    def test_hdf5_reader_exception_propagates(
        self,
        stub_loader: StubMetadataLoader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        it = RecordingIterator(
            metadata_loader=stub_loader,
            hdf5_reader=StubHDF5ReaderRaises(),
            canonical_mapper=stub_mapper,
        )
        with pytest.raises(DatasetError, match="read failure"):
            list(it.iter_recording(FAKE_RECORDING))

    def test_exception_type_is_not_wrapped(
        self,
        stub_reader: StubHDF5Reader,
        stub_mapper: StubCanonicalMapper,
    ) -> None:
        """The iterator must NOT catch-and-rethrow; the original type is preserved."""
        it = RecordingIterator(
            metadata_loader=StubMetadataLoaderRaises(),
            hdf5_reader=stub_reader,
            canonical_mapper=stub_mapper,
        )
        try:
            list(it.iter_recording(FAKE_RECORDING))
            pytest.fail("Expected DatasetError was not raised.")
        except DatasetError:
            pass  # correct — original type preserved
        except Exception as exc:
            pytest.fail(f"Exception was wrapped into unexpected type: {type(exc)}")
