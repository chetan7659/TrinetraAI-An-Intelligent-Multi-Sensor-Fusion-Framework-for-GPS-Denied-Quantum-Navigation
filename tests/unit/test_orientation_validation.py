"""Unit tests for orientation validation."""

from __future__ import annotations

import numpy as np

from trinetra.analysis.orientation_validation import validate_quaternions


def test_validate_quaternions_clean() -> None:
    # Stable orientation
    quats = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    res = validate_quaternions(quats)
    assert res["min_norm"] == 1.0
    assert res["max_norm"] == 1.0
    assert res["sign_flip_count"] == 0.0
    assert res["abnormal_jump_count"] == 0.0
    assert res["mean_angular_step_rad"] == 0.0


def test_validate_quaternions_sign_flip() -> None:
    # q and -q represent the same rotation. Dot product is -1.0.
    quats = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]
    res = validate_quaternions(quats)
    assert res["sign_flip_count"] == 1.0
    # rectified relative rotation should be 0.0
    assert res["mean_angular_step_rad"] == 0.0


def test_validate_quaternions_abnormal_jump() -> None:
    # 45 deg step (approx 0.785 rad). Exceeds 15 deg default threshold.
    q1 = [0.0, 0.0, 0.0, 1.0]
    q2 = [0.0, 0.0, np.sin(np.deg2rad(22.5)), np.cos(np.deg2rad(22.5))]
    quats = [q1, q2]
    res = validate_quaternions(quats)
    assert res["abnormal_jump_count"] == 1.0
    assert res["max_angular_step_rad"] > 0.7


def test_validate_quaternions_empty() -> None:
    res = validate_quaternions([])
    assert res["mean_norm"] == 0.0
    assert res["sign_flip_count"] == 0.0
