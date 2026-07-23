"""Generation of time-series plots for sensor streams."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from trinetra.analysis.plot_utils import AXIS_COLORS, format_axis, setup_matplotlib_style


def plot_sensor(
    sensor_name: str,
    time_axis: np.ndarray,
    data_matrix: np.ndarray,
) -> Figure:
    """Plot an individual sensor's time-series data.

    Args:
        sensor_name: The name of the sensor (e.g., 'accelerometer').
        time_axis: 1D array of timestamps in seconds.
        data_matrix: 2D array of shape (N, C) where N is number of samples and
            C is number of channels (3 for accel/gyro/mag, 4 for orientation).

    Returns:
        A matplotlib Figure containing the plotted subplots.
    """
    setup_matplotlib_style()

    num_channels = data_matrix.shape[1]

    # For 3D sensors we also plot the magnitude.
    # For Orientation (4D quaternion) we just plot W, X, Y, Z.
    plot_magnitude = num_channels == 3
    num_subplots = num_channels + (1 if plot_magnitude else 0)

    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    if num_subplots == 1:
        axes = [axes]

    # Map channels to standard labels depending on 3D vs 4D
    if num_channels == 3:
        labels = ["x", "y", "z"]
    elif num_channels == 4:
        labels = ["x", "y", "z", "w"]
    else:
        labels = [f"ch_{i}" for i in range(num_channels)]

    for i, label in enumerate(labels):
        ax = axes[i]
        color = AXIS_COLORS.get(label, "black")

        ax.plot(
            time_axis,
            data_matrix[:, i],
            color=color,
            label=f"{sensor_name} {label.upper()}",
            linewidth=1.0,
        )
        format_axis(
            ax,
            title=f"{sensor_name.title()} - {label.upper()} Axis",
            xlabel="" if i < num_subplots - 1 else "Time (s)",
            ylabel="Value",
        )

    if plot_magnitude:
        ax = axes[-1]
        magnitude = np.linalg.norm(data_matrix, axis=1)
        ax.plot(
            time_axis,
            magnitude,
            color=AXIS_COLORS["mag"],
            label=f"{sensor_name} Magnitude",
            linewidth=1.0,
        )
        format_axis(
            ax,
            title=f"{sensor_name.title()} - Magnitude",
            xlabel="Time (s)",
            ylabel="Magnitude",
        )

    fig.suptitle(f"{sensor_name.title()} Time Series", fontsize=16)
    return fig


def plot_timestamp_deltas(time_axis: np.ndarray) -> Figure:
    """Plot the differences between consecutive timestamps.

    Args:
        time_axis: 1D array of timestamps in seconds.

    Returns:
        A matplotlib Figure showing timestamp deltas.
    """
    setup_matplotlib_style()

    fig, ax = plt.subplots(1, 1)

    if len(time_axis) > 1:
        deltas = np.diff(time_axis)
        # Plot against the second timestamp onwards
        t_plot = time_axis[1:]

        ax.plot(
            t_plot,
            deltas,
            color="tab:cyan",
            marker=".",
            linestyle="none",
            markersize=2,
            label="Δt",
        )

        mean_dt = np.nanmean(deltas)
        ax.axhline(mean_dt, color="tab:red", linestyle="--", label=f"Mean Δt: {mean_dt:.4f}s")

    format_axis(
        ax,
        title="Timestamp Δt (Jitter / Dropped Frames)",
        xlabel="Time (s)",
        ylabel="Δt (s)",
    )

    return fig


def plot_composite_overview(time_axis: np.ndarray, sensors_data: dict[str, np.ndarray]) -> Figure:
    """Create a single multi-panel overview figure for all available sensors.

    Plots the magnitude of each 3D sensor stacked vertically on a shared time axis.

    Args:
        time_axis: 1D array of timestamps in seconds.
        sensors_data: A mapping from sensor name to 2D numpy arrays of data.

    Returns:
        A matplotlib Figure containing the composite plot.
    """
    setup_matplotlib_style()

    # We only include 3D sensors for magnitude plotting in the overview
    sensors_to_plot = []
    for s_name, data in sensors_data.items():
        if data.shape[1] == 3:
            sensors_to_plot.append((s_name, data))

    if not sensors_to_plot:
        # Fallback empty figure if nothing to plot
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "No 3D sensors available for composite", ha="center")
        return fig

    num_panels = len(sensors_to_plot)
    fig, axes = plt.subplots(num_panels, 1, sharex=True, figsize=(10, 2.5 * num_panels))

    if num_panels == 1:
        axes = [axes]

    for i, (sensor_name, data_matrix) in enumerate(sensors_to_plot):
        ax = axes[i]
        magnitude = np.linalg.norm(data_matrix, axis=1)

        ax.plot(
            time_axis,
            magnitude,
            color=AXIS_COLORS["mag"],
            label=f"{sensor_name.title()} Magnitude",
            linewidth=1.0,
        )

        format_axis(
            ax,
            title=f"{sensor_name.title()} Overview",
            xlabel="" if i < num_panels - 1 else "Time (s)",
            ylabel="Magnitude",
        )

    fig.suptitle("Recording Sensors Overview", fontsize=16)
    return fig
