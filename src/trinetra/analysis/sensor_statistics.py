"""Streaming aggregation for sensor descriptive statistics.

Computes dataset-wide statistics for canonical sensors:
- count
- mean
- std
- min
- max
- NaN count
- Inf count
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from trinetra.domain.interfaces.sensor_record import SensorRecord


@dataclass
class _ChannelStats:
    """Incremental statistics for a single sensor channel using Welford's algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the current mean
    min_val: float = float("inf")
    max_val: float = float("-inf")
    nan_count: int = 0
    inf_count: int = 0

    def update(self, val: float) -> None:
        if math.isnan(val):
            self.nan_count += 1
            return
        if math.isinf(val):
            self.inf_count += 1
            return

        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class SensorStatsAggregator:
    """Incrementally aggregates statistics for all canonical sensors."""

    from typing import ClassVar

    SENSORS: ClassVar[dict[str, list[str]]] = {
        "accelerometer": ["x", "y", "z"],
        "gyroscope": ["x", "y", "z"],
        "magnetometer": ["x", "y", "z"],
        "gravity": ["x", "y", "z"],
        "linear_acceleration": ["x", "y", "z"],
        "orientation": ["x", "y", "z", "w"],
    }

    def __init__(self) -> None:
        # Initialize a _ChannelStats for every sensor and channel
        self.stats: dict[str, list[_ChannelStats]] = {
            sensor: [_ChannelStats() for _ in channels] for sensor, channels in self.SENSORS.items()
        }

    def update(self, record: SensorRecord) -> None:
        """Update statistics with a single SensorRecord.

        Args:
            record: The sensor record to process.
        """
        for sensor_name in self.SENSORS:
            sensor_data = getattr(record, sensor_name, None)

            # Gracefully tolerate missing sensors
            if sensor_data is None:
                continue

            for i, val in enumerate(sensor_data):
                self.stats[sensor_name][i].update(val)

    def finalize(self) -> pd.DataFrame:
        """Finalize the aggregation and return a DataFrame of statistics.

        Returns:
            A pandas DataFrame with multi-index (Sensor, Channel) containing
            the computed descriptive statistics.
        """
        rows = []
        for sensor_name, channels in self.SENSORS.items():
            for i, channel_name in enumerate(channels):
                cs = self.stats[sensor_name][i]

                min_v = cs.min_val if cs.count > 0 else float("nan")
                max_v = cs.max_val if cs.count > 0 else float("nan")
                mean_v = cs.mean if cs.count > 0 else float("nan")
                std_v = cs.std if cs.count > 1 else float("nan")

                rows.append(
                    {
                        "Sensor": sensor_name,
                        "Channel": channel_name,
                        "count": cs.count,
                        "mean": mean_v,
                        "std": std_v,
                        "min": min_v,
                        "max": max_v,
                        "nan_count": cs.nan_count,
                        "inf_count": cs.inf_count,
                    }
                )

        df = pd.DataFrame(rows)
        df.set_index(["Sensor", "Channel"], inplace=True)
        return df
