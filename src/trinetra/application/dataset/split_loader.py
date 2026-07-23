"""Split loading orchestration service — M1.6.4.

This module provides :class:`SplitLoader`, an Application-layer service that
orchestrates iteration across entire dataset splits (e.g. "train", "seen").

Responsibility
--------------
:class:`SplitLoader` coordinates dataset discovery with single-recording
iteration. It holds references to two collaborators and sequences their calls
to produce a continuous, lazy stream of
:class:`~trinetra.domain.interfaces.sensor_record.SensorRecord` objects across
all recordings in a requested split.

It does **not**:

* read files (no ``h5py``, no JSON parsing)
* shuffle or batch data
* prefetch or use multiprocessing
* perform unit conversion or normalisation

Dependency Injection
--------------------
Collaborators are injected through the constructor. :class:`SplitLoader`
never instantiates collaborators internally, making it testable with stubs
and agnostic to specific dataset implementations.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from trinetra.domain.interfaces.sensor_record import SensorRecord

# ---------------------------------------------------------------------------
# Collaborator protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AdapterProtocol(Protocol):
    """Structural protocol for a dataset adapter.

    The concrete implementation discovers recordings on disk.
    """

    def list_recordings(self, split: str) -> list[Any]:
        """Return all recordings belonging to the requested *split*."""
        ...


@runtime_checkable
class RecordingIteratorProtocol(Protocol):
    """Structural protocol for a single-recording iterator.

    The concrete implementation is typically
    :class:`~trinetra.application.dataset.recording_iterator.RecordingIterator`.
    """

    def iter_recording(self, recording: Any) -> Iterator[SensorRecord]:
        """Yield a lazy stream of domain records for one *recording*."""
        ...


# ---------------------------------------------------------------------------
# SplitLoader
# ---------------------------------------------------------------------------


class SplitLoader:
    """Application-layer orchestration service for dataset splits.

    :class:`SplitLoader` delegates dataset discovery to an adapter, and
    single-recording iteration to a recording iterator. It acts as the
    top-level entry point for iterating over large data partitions.

    Args:
        adapter: Any object satisfying :class:`AdapterProtocol`.
            Typically :class:`~trinetra.adapters.datasets.ronin.adapter.RoninAdapter`.
        recording_iterator: Any object satisfying :class:`RecordingIteratorProtocol`.
            Typically :class:`~trinetra.application.dataset.recording_iterator.RecordingIterator`.

    Example::

        from trinetra.adapters.datasets.ronin import (
            RoninAdapter, RoninMetadataLoader,
            RoninHDF5Reader, RoninCanonicalMapper,
        )
        from trinetra.application.dataset.recording_iterator import RecordingIterator
        from trinetra.application.dataset.split_loader import SplitLoader

        adapter = RoninAdapter()
        iterator = RecordingIterator(
            metadata_loader=RoninMetadataLoader(),
            hdf5_reader=RoninHDF5Reader(),
            canonical_mapper=RoninCanonicalMapper(),
        )

        loader = SplitLoader(adapter, iterator)

        for record in loader.iter_split("train"):
            # Processes all records from all recordings in the "train" split
            pass
    """

    def __init__(
        self,
        adapter: AdapterProtocol,
        recording_iterator: RecordingIteratorProtocol,
    ) -> None:
        self._adapter = adapter
        self._recording_iterator = recording_iterator

    def iter_recordings(self, split: str) -> Iterator[Any]:
        """Yield recording objects for the given *split*.

        This is a convenience wrapper over the adapter's
        :meth:`list_recordings` method.

        Args:
            split: The partition name (e.g. "train", "seen", "unseen").

        Yields:
            Recording objects belonging to the split.
        """
        yield from self._adapter.list_recordings(split)

    def iter_split(self, split: str) -> Iterator[SensorRecord]:
        """Yield a continuous stream of domain records for an entire *split*.

        Iteration is **lazy**: recordings are processed sequentially. The
        iterator seamlessly chains the output of multiple recording
        iterators into a single flat stream.

        Args:
            split: The partition name (e.g. "train", "seen", "unseen").

        Yields:
            :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
            objects, one per time step, across all recordings in the split.

        Raises:
            Any exception raised by the collaborators propagates unchanged.
        """
        for recording in self.iter_recordings(split):
            yield from self._recording_iterator.iter_recording(recording)
