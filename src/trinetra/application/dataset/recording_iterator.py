"""Recording iteration service — M1.6.1.

This module provides :class:`RecordingIterator`, the first Application-layer
service in Trinetra-AI.

Responsibility
--------------
:class:`RecordingIterator` **orchestrates** the data-ingestion pipeline.
It holds references to three collaborators and sequences their calls to
produce a lazy stream of :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
objects for one :class:`~trinetra.adapters.datasets.ronin.models.Recording`.

It does **not**:

* open files (no ``h5py``, no ``open()``)
* parse JSON or YAML
* traverse the filesystem
* perform unit conversion, calibration, or normalisation
* filter, batch, or window frames

Pipeline
--------
::

    Recording
          │
          ▼
    MetadataLoader.load(recording)
          │
          ▼
    metadata
          │
          ▼
    HDF5Reader.read(recording, metadata)
          │
          ▼
    Iterator[RoninRawFrame]
          │
          ▼
    CanonicalMapper.map_frames(raw_frames)
          │
          ▼
    Iterator[SensorRecord]   ← yielded lazily, one frame at a time

Dependency Injection
--------------------
All collaborators are injected through the constructor.
:class:`RecordingIterator` never instantiates collaborators internally.
This makes the service trivially testable with mocks and decouples it from
any specific dataset implementation.

Collaborator protocols
----------------------
The collaborators are typed via :class:`~typing.Protocol` so that any
class satisfying the structural interface can be injected — not only the
RoNIN-specific implementations.

.. code-block:: python

    iterator = RecordingIterator(
        metadata_loader=RoninMetadataLoader(),
        hdf5_reader=RoninHDF5Reader(),
        canonical_mapper=RoninCanonicalMapper(),
    )

    for record in iterator.iter_recording(recording):
        print(record.timestamp, record.accelerometer)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol, runtime_checkable

from trinetra.domain.interfaces.sensor_record import SensorRecord

# ---------------------------------------------------------------------------
# Collaborator protocols
# ---------------------------------------------------------------------------
# Using Protocol instead of concrete adapter types keeps the Application layer
# decoupled from dataset-specific implementations.  Any class that satisfies
# the structural interface (duck-typing) is acceptable.


@runtime_checkable
class MetadataLoaderProtocol(Protocol):
    """Structural protocol for any metadata loader collaborator.

    The concrete implementation is expected to open ``recording.info_path``
    and return a metadata object.  The Application layer does not inspect
    the metadata object's fields directly.
    """

    def load(self, recording: Any) -> Any:
        """Load metadata for *recording* and return a metadata value object."""
        ...


@runtime_checkable
class RawFrameReaderProtocol(Protocol):
    """Structural protocol for any raw-frame reader collaborator.

    The concrete implementation is expected to open ``recording.hdf5_path``
    (or an equivalent binary file) and yield raw frames.
    """

    def read(self, recording: Any, metadata: Any) -> Iterator[Any]:
        """Decode *recording* using *metadata* and return a raw-frame iterator."""
        ...


@runtime_checkable
class CanonicalMapperProtocol(Protocol):
    """Structural protocol for any canonical mapper collaborator.

    The concrete implementation converts dataset-specific raw frames into
    the canonical :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
    domain type.
    """

    def map_frames(self, frames: Iterable[Any]) -> Iterator[SensorRecord]:
        """Translate *frames* into a lazy :class:`SensorRecord` iterator."""
        ...


# ---------------------------------------------------------------------------
# RecordingIterator
# ---------------------------------------------------------------------------


class RecordingIterator:
    """Application-layer orchestration service for single-recording iteration.

    :class:`RecordingIterator` wires together the three ingestion-pipeline
    components and exposes a single method — :meth:`iter_recording` — that
    yields :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
    objects one at a time without loading the entire recording into memory.

    Args:
        metadata_loader: Any object satisfying :class:`MetadataLoaderProtocol`.
            Typically :class:`~trinetra.adapters.datasets.ronin.metadata_loader.RoninMetadataLoader`.
        hdf5_reader: Any object satisfying :class:`RawFrameReaderProtocol`.
            Typically :class:`~trinetra.adapters.datasets.ronin.hdf5_reader.RoninHDF5Reader`.
        canonical_mapper: Any object satisfying :class:`CanonicalMapperProtocol`.
            Typically :class:`~trinetra.adapters.datasets.ronin.canonical_mapper.RoninCanonicalMapper`.

    Example::

        from trinetra.adapters.datasets.ronin import (
            RoninAdapter,
            RoninMetadataLoader,
            RoninHDF5Reader,
            RoninCanonicalMapper,
        )
        from trinetra.application.dataset.recording_iterator import RecordingIterator

        adapter   = RoninAdapter()
        recording = adapter.get_recording("a000_1")

        iterator  = RecordingIterator(
            metadata_loader=RoninMetadataLoader(),
            hdf5_reader=RoninHDF5Reader(),
            canonical_mapper=RoninCanonicalMapper(),
        )

        for record in iterator.iter_recording(recording):
            print(record.timestamp, record.accelerometer)
    """

    def __init__(
        self,
        metadata_loader: MetadataLoaderProtocol,
        hdf5_reader: RawFrameReaderProtocol,
        canonical_mapper: CanonicalMapperProtocol,
    ) -> None:
        self._metadata_loader = metadata_loader
        self._hdf5_reader = hdf5_reader
        self._canonical_mapper = canonical_mapper

    def iter_recording(self, recording: Any) -> Iterator[SensorRecord]:
        """Orchestrate the ingestion pipeline and yield one
        :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
        per synchronized sensor frame.

        The iteration is **lazy**: frames are decoded and mapped one at a
        time as the caller consumes the iterator.  No list is materialised
        at any stage in the pipeline.

        Pipeline steps (in order):

        1. **Metadata loading** — ``self._metadata_loader.load(recording)``
           is called *once* before iteration begins.
        2. **Raw frame reading** — ``self._hdf5_reader.read(recording, metadata)``
           returns a lazy raw-frame iterator.
        3. **Canonical mapping** — ``self._canonical_mapper.map_frames(raw_frames)``
           wraps the raw-frame iterator in a mapping generator.
        4. **Yield** — each :class:`SensorRecord` is yielded to the caller.

        Args:
            recording: A recording value object (e.g.
                :class:`~trinetra.adapters.datasets.ronin.models.Recording`)
                that the collaborators know how to handle.

        Yields:
            :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
            — one per synchronized time step, in chronological order.

        Raises:
            Any exception raised by the collaborators propagates unchanged.
            :class:`~trinetra.shared.exceptions.DatasetError` is the most
            common case (missing file, corrupted HDF5, invalid metadata).
        """
        metadata = self._metadata_loader.load(recording)
        raw_frames = self._hdf5_reader.read(recording, metadata)
        yield from self._canonical_mapper.map_frames(raw_frames)
