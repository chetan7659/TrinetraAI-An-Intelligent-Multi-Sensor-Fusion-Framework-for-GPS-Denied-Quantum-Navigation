"""Pure sampling and batching utilities for SensorRecord streams (M1.6.3).

These utilities belong to the Application layer. They operate purely on
iterables of :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
and produce lazy generators.

They prepare streams for downstream preprocessing and machine learning without
coupling to any specific dataset adapter.
"""

from __future__ import annotations

import collections
import itertools
from collections.abc import Iterable, Iterator

from trinetra.domain.interfaces.sensor_record import SensorRecord


def batch(records: Iterable[SensorRecord], batch_size: int) -> Iterator[list[SensorRecord]]:
    """Group records into non-overlapping batches.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        batch_size: The maximum number of records per batch.

    Yields:
        Lists of records of length up to ``batch_size``. The final batch
        may be smaller than ``batch_size`` if the input length is not evenly
        divisible. Evaluated lazily.

    Raises:
        ValueError: If ``batch_size <= 0``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    iterator = iter(records)
    while True:
        chunk = list(itertools.islice(iterator, batch_size))
        if not chunk:
            return
        yield chunk


def window(records: Iterable[SensorRecord], size: int) -> Iterator[tuple[SensorRecord, ...]]:
    """Create a sliding window of records.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        size: The number of records in each window.

    Yields:
        Tuples of exactly ``size`` records, evaluated lazily. If the input
        has fewer than ``size`` records, yields nothing.

    Raises:
        ValueError: If ``size <= 0``.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")

    iterator = iter(records)
    window_deque = collections.deque(itertools.islice(iterator, size), maxlen=size)

    if len(window_deque) == size:
        yield tuple(window_deque)
    else:
        # Input was shorter than window size
        return

    for record in iterator:
        window_deque.append(record)
        yield tuple(window_deque)


def stride(records: Iterable[SensorRecord], step: int) -> Iterator[SensorRecord]:
    """Yield every N-th record from the stream.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        step: The interval at which records are yielded. For example, ``step=3``
            yields the 0th, 3rd, 6th... records.

    Yields:
        Every N-th record, evaluated lazily.

    Raises:
        ValueError: If ``step <= 0``.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    yield from itertools.islice(records, 0, None, step)


def chunk_by_time(records: Iterable[SensorRecord], duration: float) -> Iterator[list[SensorRecord]]:
    """Group records into consecutive chunks covering a fixed time duration.

    The boundary of the first chunk is aligned to the ``timestamp`` of the
    first record yielded by the input stream. Subsequent chunks are bounded
    relative to that initial timestamp.

    Args:
        records: An iterable of :class:`SensorRecord` objects.
        duration: The duration (in seconds) of each chunk.

    Yields:
        Lists of records grouped by the time interval. The final chunk may
        cover a shorter duration if the stream ends. Evaluated lazily.

    Raises:
        ValueError: If ``duration <= 0``.
    """
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")

    iterator = iter(records)
    try:
        first_record = next(iterator)
    except StopIteration:
        return

    current_chunk = [first_record]
    chunk_start_time = first_record.timestamp

    for record in iterator:
        if record.timestamp - chunk_start_time >= duration:
            yield current_chunk
            # Start new chunk aligned precisely at duration multiples from the
            # original start time, so we don't drift.
            # Example: start=0, dur=5. A record at 11 goes into chunk [10, 15),
            # leaving an empty [5, 10) chunk if no records exist in that window.
            # (Note: this simple logic assumes ordered timestamps but deals with gaps
            # by pushing the start boundary forward).
            intervals_passed = int((record.timestamp - chunk_start_time) // duration)
            chunk_start_time += intervals_passed * duration
            current_chunk = [record]
        else:
            current_chunk.append(record)

    if current_chunk:
        yield current_chunk
