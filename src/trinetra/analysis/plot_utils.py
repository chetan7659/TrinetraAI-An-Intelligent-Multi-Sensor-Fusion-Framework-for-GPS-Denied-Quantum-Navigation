"""Helper utilities for publication-quality matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Canonical sensor ordering
SENSOR_ORDER = [
    "accelerometer",
    "gyroscope",
    "magnetometer",
    "gravity",
    "linear_acceleration",
    "orientation",
]

# Standard axis-to-color mapping for 3D/4D sensors
AXIS_COLORS = {
    "x": "tab:blue",
    "y": "tab:orange",
    "z": "tab:green",
    "w": "tab:purple",
    "mag": "tab:red",  # Magnitude
}


def setup_matplotlib_style() -> None:
    """Configure matplotlib defaults for publication-quality figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 8),
            "figure.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.grid": True,
        }
    )


def save_figure(fig: Figure, filepath: Path | str) -> None:
    """Save a matplotlib figure to disk safely.

    Applies tight layout and creates necessary parent directories.

    Args:
        fig: The matplotlib Figure to save.
        filepath: The destination path (e.g., .png).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_axis(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent formatting to a single matplotlib axis.

    Args:
        ax: The matplotlib AxesSubplot.
        title: The subplot title.
        xlabel: The x-axis label.
        ylabel: The y-axis label.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)
