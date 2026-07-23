"""Pure filtering utilities for SensorRecord streams (M1.6.2).

These utilities belong to the Application layer. They operate purely on
iterables of :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
and produce lazy generators.

They are completely dataset-agnostic and do not depend on the adapter layer,
HDF5 library, or filesystem.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator

from trinetra.domain.interfaces.sensor_record import SensorRecord


def filter_by_frame(
    records: Iterable[SensorRecord],
    *,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[SensorRecord]:
    """Filter records by their ``frame_id`` field.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        start: Inclusive lower bound. Records with ``frame_id < start``
            are skipped. If ``None``, no lower bound is applied.
        end: Exclusive upper bound. Records with ``frame_id >= end``
            are skipped. If ``None``, no upper bound is applied.

    Yields:
        Records satisfying the bounds, evaluated lazily.
    """
    for record in records:
        if start is not None and record.frame_id < start:
            continue
        if end is not None and record.frame_id >= end:
            continue
        yield record


def filter_by_time(
    records: Iterable[SensorRecord],
    *,
    start: float | None = None,
    end: float | None = None,
) -> Iterator[SensorRecord]:
    """Filter records by their ``timestamp`` field.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        start: Inclusive lower bound in seconds. Records with
            ``timestamp < start`` are skipped. If ``None``, no lower bound.
        end: Exclusive upper bound in seconds. Records with
            ``timestamp >= end`` are skipped. If ``None``, no upper bound.

    Yields:
        Records satisfying the bounds, evaluated lazily.
    """
    for record in records:
        if start is not None and record.timestamp < start:
            continue
        if end is not None and record.timestamp >= end:
            continue
        yield record


def take(records: Iterable[SensorRecord], count: int) -> Iterator[SensorRecord]:
    """Take the first ``count`` records from the stream.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        count: The maximum number of records to yield. If ``count <= 0``,
            yields nothing.

    Yields:
        At most ``count`` records, evaluated lazily.
    """
    if count <= 0:
        return iter([])
    yield from itertools.islice(records, count)


def skip(records: Iterable[SensorRecord], count: int) -> Iterator[SensorRecord]:
    """Skip the first ``count`` records from the stream.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        count: The number of records to skip. If ``count <= 0``, skips
            nothing and yields the entire stream.

    Yields:
        The remaining records after skipping, evaluated lazily.
    """
    if count <= 0:
        yield from records
    else:
        yield from itertools.islice(records, count, None)


def predicate_filter(
    records: Iterable[SensorRecord],
    predicate: Callable[[SensorRecord], bool],
) -> Iterator[SensorRecord]:
    """Filter records using an arbitrary predicate function.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        predicate: A callable that accepts a single :class:`SensorRecord`
            and returns ``True`` if it should be kept, ``False`` otherwise.

    Yields:
        Records for which ``predicate(record)`` is truthy, evaluated lazily.
    """
    yield from filter(predicate, records)
