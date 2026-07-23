"""Unit tests for quality metrics calculation."""

from __future__ import annotations

from trinetra.analysis.quality_metrics import (
    calculate_quality_scores,
    compute_magnitude_stats,
    detect_outliers_iqr,
    validate_quaternion_norm,
)


def test_detect_outliers_iqr_no_outliers() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert detect_outliers_iqr(data) == 0


def test_detect_outliers_iqr_with_outliers() -> None:
    # Array: 100, 100, 100, 100, 100, 500, -300
    # The IQR will be 0 (since Q1=100, Q3=100)
    # So bounds are exactly 100 to 100.
    # Therefore 500 and -300 are outliers.
    data = [100.0, 100.0, 100.0, 100.0, 100.0, 500.0, -300.0]
    assert detect_outliers_iqr(data) == 2


def test_detect_outliers_iqr_handles_nan() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, float("nan"), float("inf")]
    assert detect_outliers_iqr(data) == 0


def test_validate_quaternion_norm() -> None:
    valid_q = (0.0, 0.0, 0.0, 1.0)
    valid_q2 = (0.5, 0.5, 0.5, 0.5)  # norm is sqrt(4*0.25) = 1.0
    invalid_q = (1.0, 1.0, 1.0, 1.0)  # norm is sqrt(4) = 2.0
    nan_q = (float("nan"), 0.0, 0.0, 1.0)

    quats = [valid_q, valid_q2, invalid_q, nan_q]

    # 2 are invalid
    assert validate_quaternion_norm(quats) == 2


def test_compute_magnitude_stats() -> None:
    vectors = [
        (3.0, 4.0, 0.0),  # norm 5
        (0.0, 0.0, 0.0),  # norm 0
        (6.0, 8.0, 0.0),  # norm 10
    ]

    stats = compute_magnitude_stats(vectors)
    assert stats["mean_mag"] == 5.0
    assert stats["min_mag"] == 0.0
    assert stats["max_mag"] == 10.0
    assert stats["std_mag"] == 5.0


def test_calculate_quality_scores() -> None:
    total = 100
    missing = 10
    invalid = 5
    outliers = 5
    dropped = 2
    time_issues = 3

    scores = calculate_quality_scores(total, missing, invalid, outliers, dropped, time_issues)

    # Completeness = 1 - (10 + 2)/100 = 0.88
    assert scores["completeness_score"] == 0.88
    # Consistency = 1 - 3/100 = 0.97
    assert scores["consistency_score"] == 0.97
    # Validity = 1 - (5 + 5)/100 = 0.90
    assert scores["validity_score"] == 0.90
    # Overall = (0.88 + 0.97 + 0.90) / 3 = 0.9167
    assert scores["overall_score"] == 0.9167


def test_calculate_quality_scores_empty() -> None:
    scores = calculate_quality_scores(0, 0, 0, 0, 0, 0)
    assert scores["overall_score"] == 0.0
