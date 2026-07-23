"""Unit tests for sensor frame validation."""

from __future__ import annotations

from trinetra.analysis.frame_validation import validate_sensor_frame_consistency


def test_validate_sensor_frame_consistency_perfect() -> None:
    # A = G + L => Residual = 0
    accel = [[0.0, 0.0, 9.81], [0.1, 0.0, 9.81]]
    gravity = [[0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]
    linear = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]

    res = validate_sensor_frame_consistency(accel, gravity, linear)
    assert res["mean_residual"] == 0.0
    assert res["rms_residual"] == 0.0


def test_validate_sensor_frame_consistency_imperfect() -> None:
    # Frame 0: G + L = [0, 0, 9.81] + [0, 0, 0] = [0, 0, 9.81]. Accel = [0, 0, 10.0]. Residual norm = 0.19.
    accel = [[0.0, 0.0, 10.0]]
    gravity = [[0.0, 0.0, 9.81]]
    linear = [[0.0, 0.0, 0.0]]

    res = validate_sensor_frame_consistency(accel, gravity, linear)
    assert abs(res["mean_residual"] - 0.19) < 1e-5
    assert abs(res["rms_residual"] - 0.19) < 1e-5
    assert res["p95_residual"] == res["mean_residual"]


def test_validate_sensor_frame_consistency_empty() -> None:
    res = validate_sensor_frame_consistency([], [], [])
    assert res["rms_residual"] == 0.0
