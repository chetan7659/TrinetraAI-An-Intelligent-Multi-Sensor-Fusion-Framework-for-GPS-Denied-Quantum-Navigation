"""High-level orchestration for time-series visualization."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from trinetra.analysis.plot_utils import SENSOR_ORDER, format_axis, setup_matplotlib_style
from trinetra.analysis.summary import AnalysisResult
from trinetra.analysis.time_series import (
    plot_composite_overview,
    plot_sensor,
    plot_timestamp_deltas,
)
from trinetra.domain.interfaces.sensor_record import SensorRecord

logger = logging.getLogger(__name__)


def plot_dataset_histograms(analysis_result: AnalysisResult) -> list[Figure]:
    """Generate dataset-wide distribution histograms.

    Args:
        analysis_result: The computed analysis result containing DataFrames.

    Returns:
        A list of matplotlib Figures containing the histograms.
    """
    setup_matplotlib_style()
    figures = []

    df = analysis_result.recording_stats_df

    if df.empty:
        logger.warning("No recording stats available for histograms.")
        return figures

    # 1. Recording Duration Histogram
    fig_dur, ax_dur = plt.subplots(1, 1)
    ax_dur.hist(df["duration"], bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
    format_axis(
        ax_dur,
        title="Recording Duration Distribution",
        xlabel="Duration (s)",
        ylabel="Count",
    )
    # Remove legend from histogram if not needed, but format_axis adds one if empty.
    # We can clear the legend if no handles exist, but format_axis is safe.
    if not ax_dur.get_legend_handles_labels()[0]:
        ax_dur.get_legend().remove()

    figures.append(fig_dur)

    # 2. Sampling Frequency Histogram
    fig_freq, ax_freq = plt.subplots(1, 1)
    ax_freq.hist(
        df["sampling_frequency"], bins=30, color="tab:orange", alpha=0.7, edgecolor="black"
    )
    format_axis(
        ax_freq,
        title="Sampling Frequency Distribution",
        xlabel="Frequency (Hz)",
        ylabel="Count",
    )
    if not ax_freq.get_legend_handles_labels()[0]:
        ax_freq.get_legend().remove()

    figures.append(fig_freq)

    return figures


def generate_recording_plots(
    stream: Iterable[SensorRecord],
) -> dict[str, Figure]:
    """Buffer a single recording stream and generate all associated plots.

    Args:
        stream: An iterable of SensorRecord objects for ONE recording.

    Returns:
        A mapping from plot filename (e.g., 'accelerometer.png') to Figure.
    """
    # 1. Buffer the stream into memory (strictly ONE recording)
    timestamps = []
    sensors_data: dict[str, list[tuple[float, ...]]] = {s: [] for s in SENSOR_ORDER}

    for record in stream:
        ts = record.timestamp
        if ts is None:
            continue

        timestamps.append(ts)

        for sensor_name in SENSOR_ORDER:
            val = getattr(record, sensor_name, None)
            if val is not None:
                sensors_data[sensor_name].append(val)

    if not timestamps:
        logger.warning("Empty stream provided, skipping plots.")
        return {}

    t_arr = np.array(timestamps)

    # Filter to only sensors that have data, and convert to numpy arrays
    valid_sensors: dict[str, np.ndarray] = {}
    for s_name in SENSOR_ORDER:
        s_data = sensors_data[s_name]
        if s_data and len(s_data) == len(t_arr):
            valid_sensors[s_name] = np.array(s_data)

    # 2. Generate plots
    plots: dict[str, Figure] = {}

    # Timestamp deltas
    plots["timestamp_delta.png"] = plot_timestamp_deltas(t_arr)

    # Individual sensors
    for sensor_name in SENSOR_ORDER:
        if sensor_name in valid_sensors:
            data_matrix = valid_sensors[sensor_name]
            plots[f"{sensor_name}.png"] = plot_sensor(sensor_name, t_arr, data_matrix)

    # Composite overview
    plots["composite_overview.png"] = plot_composite_overview(t_arr, valid_sensors)

    return plots
