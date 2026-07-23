"""Unit tests for dataset-wide statistics."""

from __future__ import annotations

import pandas as pd

from trinetra.analysis.dataset_statistics import (
    compute_dataset_statistics,
    compute_recordings_per_split,
)


def test_compute_dataset_statistics_empty() -> None:
    result = compute_dataset_statistics([])
    assert result["total_recordings"] == 0
    assert result["total_splits"] == 0
    assert result["total_frames"] == 0
    assert result["total_duration"] == 0.0
    assert result["min_duration"] == 0.0
    assert result["max_duration"] == 0.0
    assert result["avg_duration"] == 0.0
    assert result["avg_sampling_frequency"] == 0.0


def test_compute_dataset_statistics_values() -> None:
    stats = [
        {
            "recording_id": "r1",
            "split": "train",
            "frame_count": 100,
            "duration": 10.0,
            "sampling_frequency": 10.0,
        },
        {
            "recording_id": "r2",
            "split": "train",
            "frame_count": 200,
            "duration": 20.0,
            "sampling_frequency": 10.0,
        },
        {
            "recording_id": "r3",
            "split": "test",
            "frame_count": 300,
            "duration": 15.0,
            "sampling_frequency": 20.0,
        },
    ]

    result = compute_dataset_statistics(stats)

    assert result["total_recordings"] == 3
    assert result["total_splits"] == 2
    assert result["total_frames"] == 600
    assert result["total_duration"] == 45.0
    assert result["min_duration"] == 10.0
    assert result["max_duration"] == 20.0
    assert result["avg_duration"] == 15.0  # (10+20+15)/3
    assert result["avg_sampling_frequency"] == 13.333333333333334  # (10+10+20)/3


def test_compute_recordings_per_split() -> None:
    stats = [
        {"recording_id": "r1", "split": "train"},
        {"recording_id": "r2", "split": "train"},
        {"recording_id": "r3", "split": "test"},
    ]

    result = compute_recordings_per_split(stats)
    assert len(result) == 2
    assert result["train"] == 2
    assert result["test"] == 1


def test_compute_recordings_per_split_empty() -> None:
    result = compute_recordings_per_split([])
    assert len(result) == 0
    assert isinstance(result, pd.Series)
