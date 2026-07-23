"""Unit tests for EDA serialization."""

from __future__ import annotations

import pandas as pd

from trinetra.analysis.report_export import export_reports
from trinetra.analysis.summary import AnalysisResult


def test_export_reports(tmp_path) -> None:
    # Create synthetic AnalysisResult
    dataset_stats = pd.Series(
        {
            "total_recordings": 2,
            "total_splits": 1,
            "total_frames": 100,
            "total_duration": 10.0,
            "min_duration": 5.0,
            "max_duration": 5.0,
            "avg_duration": 5.0,
            "avg_sampling_frequency": 10.0,
        }
    )

    recordings_per_split = pd.Series({"train": 2})

    recording_stats_df = pd.DataFrame(
        [
            {"recording_id": "r1", "split": "train"},
            {"recording_id": "r2", "split": "train"},
        ]
    )

    timestamp_stats_df = pd.DataFrame(
        [
            {"recording_id": "r1", "is_monotonic": True, "duplicate_count": 0, "missing_count": 0},
            {"recording_id": "r2", "is_monotonic": False, "duplicate_count": 1, "missing_count": 0},
        ]
    )

    sensor_stats_df = pd.DataFrame(
        [
            {"Sensor": "accelerometer", "Channel": "x", "count": 100, "mean": 0.0},
        ]
    ).set_index(["Sensor", "Channel"])

    result = AnalysisResult(
        dataset_stats=dataset_stats,
        recordings_per_split=recordings_per_split,
        recording_stats_df=recording_stats_df,
        timestamp_stats_df=timestamp_stats_df,
        sensor_stats_df=sensor_stats_df,
    )

    export_dir = tmp_path / "reports"

    export_reports(result, export_dir)

    assert export_dir.exists()
    assert (export_dir / "dataset_statistics.csv").exists()
    assert (export_dir / "recording_statistics.csv").exists()
    assert (export_dir / "timestamp_statistics.csv").exists()
    assert (export_dir / "sensor_statistics.csv").exists()

    summary_file = export_dir / "summary.md"
    assert summary_file.exists()

    content = summary_file.read_text(encoding="utf-8")
    assert "**Total Recordings**: 2" in content
    assert "**train**: 2 recordings" in content
    assert "Recordings with non-monotonic timestamps: 1" in content
    assert "Recordings with duplicated timestamps: 1" in content
