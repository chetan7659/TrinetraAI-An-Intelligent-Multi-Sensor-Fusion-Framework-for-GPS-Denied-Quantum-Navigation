"""Pure computation functions for detecting anomalies and assessing sensor quality."""

from __future__ import annotations

import math
from collections.abc import Sequence


def detect_outliers_iqr(values: Sequence[float], iqr_multiplier: float = 1.5) -> int:
    """Detect the number of outliers using the IQR rule.

    Args:
        values: A sequence of numeric values.
        iqr_multiplier: The multiplier for the IQR threshold (default 1.5).

    Returns:
        The number of outliers detected.
    """
    if not values:
        return 0

    # Filter out NaNs
    valid_vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not valid_vals:
        return 0

    sorted_vals = sorted(valid_vals)
    n = len(sorted_vals)

    if n < 4:
        return 0

    def get_percentile(data: list[float], percentile: float) -> float:
        k = (len(data) - 1) * percentile
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1

    q1 = get_percentile(sorted_vals, 0.25)
    q3 = get_percentile(sorted_vals, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)

    outliers = sum(1 for v in sorted_vals if v < lower_bound or v > upper_bound)
    return outliers


def validate_quaternion_norm(
    quaternions: Sequence[tuple[float, float, float, float]], tol: float = 1e-3
) -> int:
    """Count how many quaternions deviate from a norm of 1.0 beyond tolerance.

    Args:
        quaternions: Sequence of (x, y, z, w) tuples.
        tol: Tolerance for the norm deviation.

    Returns:
        The number of invalid quaternions.
    """
    invalid_count = 0
    for q in quaternions:
        if not q or len(q) != 4:
            continue
        try:
            if any(math.isnan(v) for v in q):
                invalid_count += 1
                continue

            norm = math.sqrt(sum(v**2 for v in q))
            if abs(norm - 1.0) > tol:
                invalid_count += 1
        except (ValueError, TypeError):
            # E.g. invalid types
            invalid_count += 1
    return invalid_count


def compute_magnitude_stats(vectors: Sequence[Sequence[float]]) -> dict[str, float]:
    """Compute mean, std, min, and max magnitude for a sequence of vectors.

    Args:
        vectors: A sequence of N-dimensional vectors.

    Returns:
        Dictionary with 'mean_mag', 'std_mag', 'min_mag', 'max_mag'.
    """
    if not vectors:
        return {
            "mean_mag": float("nan"),
            "std_mag": float("nan"),
            "min_mag": float("nan"),
            "max_mag": float("nan"),
        }

    mags = []
    for v in vectors:
        if any(math.isnan(x) or math.isinf(x) for x in v):
            continue
        mags.append(math.sqrt(sum(x**2 for x in v)))

    if not mags:
        return {
            "mean_mag": float("nan"),
            "std_mag": float("nan"),
            "min_mag": float("nan"),
            "max_mag": float("nan"),
        }

    n = len(mags)
    mean_mag = sum(mags) / n
    min_mag = min(mags)
    max_mag = max(mags)

    if n > 1:
        variance = sum((m - mean_mag) ** 2 for m in mags) / (n - 1)
        std_mag = math.sqrt(variance)
    else:
        std_mag = 0.0

    return {
        "mean_mag": mean_mag,
        "std_mag": std_mag,
        "min_mag": min_mag,
        "max_mag": max_mag,
    }


def calculate_quality_scores(
    total_frames: int,
    missing_samples: int,
    invalid_samples: int,
    outliers: int,
    dropped_frames: int,
    timestamp_issues: int,
) -> dict[str, float]:
    """Calculate completeness, consistency, validity, and overall quality scores in [0, 1].

    Args:
        total_frames: Total number of frames in the recording.
        missing_samples: Count of missing or NaN samples.
        invalid_samples: Count of invalid values (e.g. Inf, constant, saturated, invalid norm).
        outliers: Total number of identified outliers across all sensors.
        dropped_frames: Count of significant gaps in timestamps.
        timestamp_issues: Non-monotonic or duplicated timestamps.

    Returns:
        Dictionary with 'completeness_score', 'consistency_score', 'validity_score', and 'overall_score'.
    """
    if total_frames <= 0:
        return {
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "validity_score": 0.0,
            "overall_score": 0.0,
        }

    # Completeness: Penalizes missing data and dropped frames
    completeness_penalty = (missing_samples + dropped_frames) / total_frames
    completeness_score = max(0.0, 1.0 - completeness_penalty)

    # Consistency: Penalizes temporal issues (jitter, duplicates, non-monotonic)
    consistency_penalty = timestamp_issues / total_frames
    consistency_score = max(0.0, 1.0 - consistency_penalty)

    # Validity: Penalizes incorrect/unrealistic values (outliers, saturation, inf)
    validity_penalty = (invalid_samples + outliers) / total_frames
    validity_score = max(0.0, 1.0 - validity_penalty)

    # Overall is the unweighted average
    overall_score = (completeness_score + consistency_score + validity_score) / 3.0

    return {
        "completeness_score": round(completeness_score, 4),
        "consistency_score": round(consistency_score, 4),
        "validity_score": round(validity_score, 4),
        "overall_score": round(overall_score, 4),
    }
