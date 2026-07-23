"""Computation of dataset-wide statistics.

Aggregates per-recording metrics into overall dataset statistics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_dataset_statistics(recording_stats: list[dict[str, Any]]) -> pd.Series:
    """Compute dataset-wide statistics from a list of recording statistics.

    Args:
        recording_stats: A list of dictionaries, where each dictionary
            was produced by RecordingStatsAggregator.finalize().

    Returns:
        A pandas Series containing overall dataset metrics.
    """
    if not recording_stats:
        return pd.Series(
            {
                "total_recordings": 0,
                "total_splits": 0,
                "total_frames": 0,
                "total_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "avg_duration": 0.0,
                "avg_sampling_frequency": 0.0,
            }
        )

    df = pd.DataFrame(recording_stats)

    total_recordings = len(df)
    total_splits = df["split"].nunique()
    total_frames = df["frame_count"].sum()
    total_duration = df["duration"].sum()

    # Exclude zero duration recordings from min/max/avg duration if necessary,
    # but strictly following requirements:
    valid_durations = df[df["duration"] > 0]["duration"]
    if not valid_durations.empty:
        min_duration = valid_durations.min()
        max_duration = valid_durations.max()
        avg_duration = valid_durations.mean()
    else:
        min_duration = 0.0
        max_duration = 0.0
        avg_duration = 0.0

    valid_freqs = df[df["sampling_frequency"] > 0]["sampling_frequency"]
    avg_sampling_frequency = valid_freqs.mean() if not valid_freqs.empty else 0.0

    return pd.Series(
        {
            "total_recordings": total_recordings,
            "total_splits": total_splits,
            "total_frames": total_frames,
            "total_duration": total_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "avg_duration": avg_duration,
            "avg_sampling_frequency": avg_sampling_frequency,
        }
    )


def compute_recordings_per_split(recording_stats: list[dict[str, Any]]) -> pd.Series:
    """Compute the number of recordings in each split.

    Args:
        recording_stats: A list of dictionaries, where each dictionary
            was produced by RecordingStatsAggregator.finalize().

    Returns:
        A pandas Series indexed by split name containing recording counts.
    """
    if not recording_stats:
        return pd.Series(dtype=int)

    df = pd.DataFrame(recording_stats)
    return df["split"].value_counts()
