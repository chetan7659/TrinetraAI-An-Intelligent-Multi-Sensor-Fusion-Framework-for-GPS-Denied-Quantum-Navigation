"""Unit tests for high-level visualization orchestration."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from trinetra.analysis.summary import AnalysisResult
from trinetra.analysis.visualization import (
    generate_recording_plots,
    plot_dataset_histograms,
)
from trinetra.domain.interfaces.sensor_record import SensorRecord


def test_plot_dataset_histograms() -> None:
    # Synthetic AnalysisResult
    result = AnalysisResult(
        dataset_stats=pd.Series(),
        recordings_per_split=pd.Series(),
        recording_stats_df=pd.DataFrame(
            [
                {"duration": 10.0, "sampling_frequency": 100.0},
                {"duration": 15.0, "sampling_frequency": 200.0},
            ]
        ),
        timestamp_stats_df=pd.DataFrame(),
        sensor_stats_df=pd.DataFrame(),
    )

    figures = plot_dataset_histograms(result)

    assert len(figures) == 2
    assert isinstance(figures[0], Figure)
    assert isinstance(figures[1], Figure)


def test_plot_dataset_histograms_empty() -> None:
    result = AnalysisResult(
        dataset_stats=pd.Series(),
        recordings_per_split=pd.Series(),
        recording_stats_df=pd.DataFrame(),
        timestamp_stats_df=pd.DataFrame(),
        sensor_stats_df=pd.DataFrame(),
    )

    figures = plot_dataset_histograms(result)
    assert len(figures) == 0


def test_generate_recording_plots() -> None:
    stream = [
        SensorRecord(
            frame_id=0,
            timestamp=0.0,
            accelerometer=(0.0, 0.0, 0.0),
            gyroscope=(0.0, 0.0, 0.0),
            magnetometer=(0.0, 0.0, 0.0),
            gravity=None,  # Missing sensor
            linear_acceleration=None,
            orientation=(0.0, 0.0, 0.0, 1.0),
        ),
        SensorRecord(
            frame_id=1,
            timestamp=1.0,
            accelerometer=(1.0, 1.0, 1.0),
            gyroscope=(1.0, 1.0, 1.0),
            magnetometer=(1.0, 1.0, 1.0),
            gravity=None,
            linear_acceleration=None,
            orientation=(1.0, 1.0, 1.0, 1.0),
        ),
    ]

    plots = generate_recording_plots(stream)

    # Expect: timestamp_delta, composite, and 4 sensors (accel, gyro, mag, orient)
    assert len(plots) == 6

    expected_files = [
        "timestamp_delta.png",
        "composite_overview.png",
        "accelerometer.png",
        "gyroscope.png",
        "magnetometer.png",
        "orientation.png",
    ]

    for fname in expected_files:
        assert fname in plots
        assert isinstance(plots[fname], Figure)

    assert "gravity.png" not in plots
    assert "linear_acceleration.png" not in plots


def test_generate_recording_plots_empty_stream() -> None:
    plots = generate_recording_plots([])
    assert len(plots) == 0
