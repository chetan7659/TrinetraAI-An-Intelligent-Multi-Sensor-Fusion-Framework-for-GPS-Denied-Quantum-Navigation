"""Unit tests for plot utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt

from trinetra.analysis.plot_utils import (
    AXIS_COLORS,
    SENSOR_ORDER,
    format_axis,
    save_figure,
    setup_matplotlib_style,
)


def test_setup_matplotlib_style() -> None:
    # Ensure it doesn't crash and sets keys
    setup_matplotlib_style()
    assert plt.rcParams["figure.figsize"] == [10.0, 8.0]
    assert plt.rcParams["figure.dpi"] == 300.0


def test_save_figure(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    out_path = tmp_path / "subdir" / "test_fig.png"
    save_figure(fig, out_path)

    assert out_path.exists()
    assert out_path.is_file()


def test_format_axis() -> None:
    fig, ax = plt.subplots()
    ax.plot([1], [1], label="test")
    format_axis(ax, title="My Title", xlabel="Time", ylabel="Val")

    assert ax.get_title() == "My Title"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Val"

    plt.close(fig)


def test_constants_exist() -> None:
    assert isinstance(SENSOR_ORDER, list)
    assert len(SENSOR_ORDER) >= 6
    assert "accelerometer" in SENSOR_ORDER

    assert isinstance(AXIS_COLORS, dict)
    assert "x" in AXIS_COLORS
    assert "y" in AXIS_COLORS
    assert "z" in AXIS_COLORS
