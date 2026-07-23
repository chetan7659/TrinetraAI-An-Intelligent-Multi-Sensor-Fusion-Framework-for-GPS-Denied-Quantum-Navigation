"""Single-pass streaming analysis for assessing sensor quality."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from trinetra.analysis.plot_utils import SENSOR_ORDER
from trinetra.analysis.quality_metrics import (
    calculate_quality_scores,
    compute_magnitude_stats,
    detect_outliers_iqr,
    validate_quaternion_norm,
)
from trinetra.domain.interfaces.sensor_record import SensorRecord


def assess_recording_quality(recording_id: str, stream: Iterable[SensorRecord]) -> dict[str, Any]:
    """Assess the sensor quality of a single recording stream.

    Processes the recording sequentially and computes missing samples, invalid
    samples, outliers, timestamp integrity, and sensor magnitude statistics.

    Args:
        recording_id: Unique identifier for the recording.
        stream: An iterable of SensorRecord objects for ONE recording.

    Returns:
        A dictionary containing quality metrics and scores.
    """
    total_frames = 0
    timestamps: list[float] = []

    # Track raw values for outlier detection and magnitude analysis
    sensor_data: dict[str, list[tuple[float, ...]]] = {s: [] for s in SENSOR_ORDER}

    # Counts of structural anomalies
    missing_samples = {s: 0 for s in SENSOR_ORDER}
    nan_count = {s: 0 for s in SENSOR_ORDER}
    inf_count = {s: 0 for s in SENSOR_ORDER}
    zero_vectors = {s: 0 for s in SENSOR_ORDER}
    constant_values = {s: 0 for s in SENSOR_ORDER}
    saturation_count = {s: 0 for s in SENSOR_ORDER}

    # Saturation thresholds
    saturation_limits = {
        "accelerometer": 156.96,  # ~16g
        "linear_acceleration": 156.96,  # ~16g
        "gyroscope": 34.9,  # ~2000 deg/s
    }

    # For constant value detection
    previous_values: dict[str, tuple[float, ...]] = {}

    for record in stream:
        total_frames += 1

        if record.timestamp is not None:
            timestamps.append(record.timestamp)

        for s_name in SENSOR_ORDER:
            val = getattr(record, s_name, None)
            if val is None:
                missing_samples[s_name] += 1
                continue

            sensor_data[s_name].append(val)

            # Check NaN / Inf / Zeros
            has_nan = any(math.isnan(x) for x in val)
            has_inf = any(math.isinf(x) for x in val)
            is_zero = all(x == 0.0 for x in val)

            if has_nan:
                nan_count[s_name] += 1
            if has_inf:
                inf_count[s_name] += 1
            if is_zero and not has_nan and not has_inf:
                zero_vectors[s_name] += 1

            # Check Saturation
            limit = saturation_limits.get(s_name)
            if (
                limit is not None
                and not has_nan
                and not has_inf
                and any(abs(x) >= limit for x in val)
            ):
                saturation_count[s_name] += 1

            # Check Constant Values
            prev = previous_values.get(s_name)
            if prev is not None and prev == val and not has_nan and not has_inf:
                constant_values[s_name] += 1

            previous_values[s_name] = val

    # Compute Temporal Consistency
    dropped_frames = 0
    duplicate_timestamps = 0
    non_monotonic = 0
    max_gap = 0.0
    freq_variance = 0.0

    if len(timestamps) > 1:
        deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        mean_dt = sum(deltas) / len(deltas)
        max_gap = max(deltas)

        for dt in deltas:
            if dt == 0:
                duplicate_timestamps += 1
            elif dt < 0:
                non_monotonic += 1
            elif dt > mean_dt * 2.0:  # Simple heuristic for dropped frame
                dropped_frames += 1

        freq_variance = sum((dt - mean_dt) ** 2 for dt in deltas) / len(deltas)

    timestamp_issues = duplicate_timestamps + non_monotonic

    # Compute Statistical Consistency & Outliers
    outlier_counts = {s: 0 for s in SENSOR_ORDER}
    invalid_quaternions = 0
    mag_stats = {}

    total_invalid_samples = 0
    total_missing = 0
    total_outliers = 0

    for s_name in SENSOR_ORDER:
        data = sensor_data[s_name]
        if not data:
            continue

        # Outliers per axis
        num_channels = len(data[0])
        s_outliers = 0
        for i in range(num_channels):
            channel_data = [row[i] for row in data]
            s_outliers += detect_outliers_iqr(channel_data)
        outlier_counts[s_name] = s_outliers
        total_outliers += s_outliers

        # Orientations
        if s_name == "orientation":
            invalid_quaternions = validate_quaternion_norm(data)
            total_invalid_samples += invalid_quaternions

        # Magnitude
        if num_channels == 3:
            mag_stats[s_name] = compute_magnitude_stats(data)

        total_missing += missing_samples[s_name]
        total_invalid_samples += (
            nan_count[s_name]
            + inf_count[s_name]
            + constant_values[s_name]
            + saturation_count[s_name]
        )

    scores = calculate_quality_scores(
        total_frames=total_frames,
        missing_samples=total_missing,
        invalid_samples=total_invalid_samples,
        outliers=total_outliers,
        dropped_frames=dropped_frames,
        timestamp_issues=timestamp_issues,
    )

    return {
        "recording_id": recording_id,
        "total_frames": total_frames,
        "missing_samples": missing_samples,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vectors": zero_vectors,
        "constant_values": constant_values,
        "saturation_count": saturation_count,
        "outlier_counts": outlier_counts,
        "temporal_consistency": {
            "dropped_frames": dropped_frames,
            "duplicate_timestamps": duplicate_timestamps,
            "non_monotonic": non_monotonic,
            "max_gap": max_gap,
            "freq_variance": freq_variance,
        },
        "invalid_quaternions": invalid_quaternions,
        "magnitude_stats": mag_stats,
        "scores": scores,
    }
