"""Streaming aggregation for timestamp integrity statistics.

Verifies:
- timestamps monotonic
- duplicate timestamps
- missing timestamps
- minimum Δt
- maximum Δt
- average Δt
"""

from __future__ import annotations

from typing import Any

from trinetra.domain.interfaces.sensor_record import SensorRecord


class TimestampStatsAggregator:
    """Incrementally aggregates timestamp statistics for a single recording."""

    def __init__(self, recording_id: str, split_name: str) -> None:
        """Initialize the aggregator.

        Args:
            recording_id: The ID of the recording being processed.
            split_name: The dataset split (e.g. 'train') this recording belongs to.
        """
        self.recording_id = recording_id
        self.split_name = split_name
        self.frame_count = 0

        self.previous_timestamp: float | None = None

        self.is_monotonic = True
        self.duplicate_count = 0
        self.missing_count = 0  # Assuming missing timestamps manifest as gaps or NaNs

        self.min_dt: float = float("inf")
        self.max_dt: float = float("-inf")
        self.sum_dt: float = 0.0
        self.dt_count = 0

    def update(self, record: SensorRecord) -> None:
        """Update statistics with a single SensorRecord.

        Args:
            record: The sensor record to process.
        """
        self.frame_count += 1
        current_ts = record.timestamp

        if current_ts is None:
            self.missing_count += 1
            return

        if self.previous_timestamp is not None:
            dt = current_ts - self.previous_timestamp

            if dt < 0:
                self.is_monotonic = False
            elif dt == 0:
                self.duplicate_count += 1
            else:
                self.min_dt = min(self.min_dt, dt)
                self.max_dt = max(self.max_dt, dt)
                self.sum_dt += dt
                self.dt_count += 1

        self.previous_timestamp = current_ts

    def finalize(self) -> dict[str, Any]:
        """Finalize the aggregation and return a dictionary of statistics.

        Returns:
            A dictionary containing the computed statistics.
        """
        avg_dt = self.sum_dt / self.dt_count if self.dt_count > 0 else 0.0

        min_dt_val = self.min_dt if self.min_dt != float("inf") else 0.0
        max_dt_val = self.max_dt if self.max_dt != float("-inf") else 0.0

        return {
            "recording_id": self.recording_id,
            "split": self.split_name,
            "frame_count": self.frame_count,
            "is_monotonic": self.is_monotonic,
            "duplicate_count": self.duplicate_count,
            "missing_count": self.missing_count,
            "min_dt": min_dt_val,
            "max_dt": max_dt_val,
            "avg_dt": avg_dt,
        }
