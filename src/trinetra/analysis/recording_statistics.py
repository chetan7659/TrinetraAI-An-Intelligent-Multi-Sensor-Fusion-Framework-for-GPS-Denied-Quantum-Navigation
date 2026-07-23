"""Streaming aggregation for recording-level statistics.

Computes:
- frame count
- duration
- sampling frequency
- start/end timestamps
"""

from __future__ import annotations

from typing import Any

from trinetra.domain.interfaces.sensor_record import SensorRecord


class RecordingStatsAggregator:
    """Incrementally aggregates statistics for a single recording."""

    def __init__(self, recording_id: str, split_name: str) -> None:
        """Initialize the aggregator.

        Args:
            recording_id: The ID of the recording being processed.
            split_name: The dataset split (e.g. 'train') this recording belongs to.
        """
        self.recording_id = recording_id
        self.split_name = split_name
        self.frame_count = 0
        self.start_timestamp: float | None = None
        self.end_timestamp: float | None = None

    def update(self, record: SensorRecord) -> None:
        """Update statistics with a single SensorRecord.

        Args:
            record: The sensor record to process.
        """
        self.frame_count += 1
        if self.start_timestamp is None:
            self.start_timestamp = record.timestamp
        self.end_timestamp = record.timestamp

    def finalize(self) -> dict[str, Any]:
        """Finalize the aggregation and return a dictionary of statistics.

        Returns:
            A dictionary containing the computed statistics.
        """
        duration = 0.0
        sampling_frequency = 0.0

        if (
            self.frame_count > 0
            and self.start_timestamp is not None
            and self.end_timestamp is not None
        ):
            duration = self.end_timestamp - self.start_timestamp
            if duration > 0.0:
                # We have N frames spanning a duration.
                # Frequency is defined as (frame_count - 1) / duration for monotonic intervals,
                # but commonly frequency is total frames / duration (or we can use frame_count / duration
                # if duration isn't exactly the interval between first and last, but here it is).
                # To be consistent with standard rate calculations: (count) / duration, or
                # since duration is exactly (end - start), the average interval is duration / (count - 1).
                # Let's use count / duration if duration > 0 for overall rate, or count - 1.
                # Standard is usually just total items / total time.
                sampling_frequency = self.frame_count / duration

        return {
            "recording_id": self.recording_id,
            "split": self.split_name,
            "frame_count": self.frame_count,
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }
