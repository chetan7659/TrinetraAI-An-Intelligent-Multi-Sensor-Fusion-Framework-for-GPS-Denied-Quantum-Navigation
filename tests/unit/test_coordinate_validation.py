"""Unit tests for coordinate validation."""

from __future__ import annotations

from trinetra.analysis.coordinate_validation import check_coordinate_consistency, validate_gravity


def test_validate_gravity_success() -> None:
    # Gravity is exactly 9.81 m/s^2 along Z
    vectors = [[0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]
    res = validate_gravity(vectors)
    assert abs(res["mean_magnitude"] - 9.81) < 1e-5
    assert abs(res["bias"]) < 1e-5
    assert res["std_magnitude"] == 0.0


def test_validate_gravity_empty() -> None:
    res = validate_gravity([])
    assert res["mean_magnitude"] == 0.0
    assert res["bias"] == 0.0


def test_check_coordinate_consistency() -> None:
    accel = [[0.1, 0.2, 9.8], [0.1, 0.2, 9.9]]
    gravity = [[0.0, 0.0, 9.8], [0.0, 0.0, 9.81]]
    res = check_coordinate_consistency(accel, gravity)
    # Z axis has max value for gravity (index 2)
    assert res["gravity_alignment_axis"] == 2
    assert res["axis_polarities_consistent"] == 1.0


def test_check_coordinate_consistency_polarity_flip() -> None:
    accel = [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]
    gravity = [[0.0, 0.0, 9.81], [0.0, 0.0, -9.81]]
    res = check_coordinate_consistency(accel, gravity)
    assert res["axis_polarities_consistent"] == 0.0
