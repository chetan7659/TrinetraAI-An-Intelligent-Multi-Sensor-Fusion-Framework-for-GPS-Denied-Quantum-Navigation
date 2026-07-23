"""Unit tests for sensor statistics."""

from __future__ import annotations

import pandas as pd

from trinetra.analysis.sensor_statistics import SensorStatsAggregator
from trinetra.domain.interfaces.sensor_record import SensorRecord


def _make_record(
    accel: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    gyro: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
) -> SensorRecord:
    return SensorRecord(
        frame_id=0,
        timestamp=0.0,
        accelerometer=accel,  # type: ignore
        gyroscope=gyro,  # type: ignore
        linear_acceleration=None,  # missing sensor test
        gravity=None,
        orientation=None,
        magnetometer=None,
    )


def test_sensor_stats_empty() -> None:
    agg = SensorStatsAggregator()
    df = agg.finalize()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3 * 5 + 4  # 5 3D sensors + 1 4D sensor = 19 channels

    # All counts should be 0
    assert (df["count"] == 0).all()
    # Min/max/mean/std should be NaN
    assert df["min"].isna().all()
    assert df["max"].isna().all()


def test_sensor_stats_values() -> None:
    agg = SensorStatsAggregator()

    agg.update(_make_record(accel=(1.0, 2.0, 3.0)))
    agg.update(_make_record(accel=(3.0, 4.0, 5.0)))
    agg.update(_make_record(accel=(5.0, 6.0, 7.0)))

    df = agg.finalize()

    accel_x = df.loc[("accelerometer", "x")]
    assert accel_x["count"] == 3
    assert accel_x["min"] == 1.0
    assert accel_x["max"] == 5.0
    assert accel_x["mean"] == 3.0
    assert (
        accel_x["std"] == 2.0
    )  # sqrt( ((1-3)^2 + (3-3)^2 + (5-3)^2) / 2 ) = sqrt( (4+0+4)/2 ) = 2.0

    accel_y = df.loc[("accelerometer", "y")]
    assert accel_y["mean"] == 4.0
    assert accel_y["std"] == 2.0


def test_sensor_stats_nan_inf_handling() -> None:
    agg = SensorStatsAggregator()

    agg.update(_make_record(accel=(1.0, float("nan"), float("inf"))))
    agg.update(_make_record(accel=(3.0, 4.0, float("-inf"))))

    df = agg.finalize()

    accel_x = df.loc[("accelerometer", "x")]
    assert accel_x["count"] == 2
    assert accel_x["nan_count"] == 0
    assert accel_x["inf_count"] == 0

    accel_y = df.loc[("accelerometer", "y")]
    assert accel_y["count"] == 1
    assert accel_y["nan_count"] == 1
    assert accel_y["mean"] == 4.0

    accel_z = df.loc[("accelerometer", "z")]
    assert accel_z["count"] == 0
    assert accel_z["inf_count"] == 2


def test_sensor_stats_missing_sensors() -> None:
    agg = SensorStatsAggregator()

    # Passing None for sensors tests the graceful omission
    agg.update(_make_record(accel=None, gyro=None))

    df = agg.finalize()
    assert (df["count"] == 0).all()
