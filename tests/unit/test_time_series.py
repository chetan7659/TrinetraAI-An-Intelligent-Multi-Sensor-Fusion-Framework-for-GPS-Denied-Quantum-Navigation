"""Unit tests for time-series plotting functions."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from trinetra.analysis.time_series import (
    plot_composite_overview,
    plot_sensor,
    plot_timestamp_deltas,
)


def test_plot_sensor_3d() -> None:
    time_axis = np.array([0.0, 1.0, 2.0])
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )

    fig = plot_sensor("accelerometer", time_axis, data)

    assert isinstance(fig, Figure)
    # 3 subplots for XYZ + 1 for magnitude
    assert len(fig.axes) == 4


def test_plot_sensor_4d() -> None:
    time_axis = np.array([0.0, 1.0, 2.0])
    data = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.3, 0.9],
            [0.2, 0.4, 0.6, 0.8],
        ]
    )

    fig = plot_sensor("orientation", time_axis, data)

    assert isinstance(fig, Figure)
    # 4 subplots for XYZW (no magnitude for orientation)
    assert len(fig.axes) == 4


def test_plot_timestamp_deltas() -> None:
    time_axis = np.array([0.0, 0.1, 0.2, 0.4, 0.5])

    fig = plot_timestamp_deltas(time_axis)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_composite_overview() -> None:
    time_axis = np.array([0.0, 1.0])

    # Only 3D sensors are included in the overview
    sensors_data = {
        "accelerometer": np.array([[1, 0, 0], [2, 0, 0]]),
        "gyroscope": np.array([[0, 1, 0], [0, 2, 0]]),
        "orientation": np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
    }

    fig = plot_composite_overview(time_axis, sensors_data)

    assert isinstance(fig, Figure)
    # Should only plot accelerometer and gyroscope magnitudes
    assert len(fig.axes) == 2


def test_plot_composite_overview_empty() -> None:
    time_axis = np.array([0.0, 1.0])
    sensors_data = {}

    fig = plot_composite_overview(time_axis, sensors_data)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
