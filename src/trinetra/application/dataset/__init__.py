"""Application-layer dataset services.

This package contains application services that orchestrate the data
ingestion pipeline components defined in:

* ``trinetra.adapters.datasets`` (adapters)
* ``trinetra.domain.interfaces`` (domain contracts)

Services in this package coordinate collaborators — they do NOT perform
parsing, decoding, mapping, preprocessing, or feature engineering.
"""

from .filters import filter_by_frame, filter_by_time, predicate_filter, skip, take
from .recording_iterator import RecordingIterator
from .samplers import batch, chunk_by_time, stride, window

__all__ = [
    "RecordingIterator",
    "batch",
    "chunk_by_time",
    "filter_by_frame",
    "filter_by_time",
    "predicate_filter",
    "skip",
    "stride",
    "take",
    "window",
]
