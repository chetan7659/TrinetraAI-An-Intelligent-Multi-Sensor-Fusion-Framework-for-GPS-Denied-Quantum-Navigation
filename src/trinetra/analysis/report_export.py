"""Serialization of EDA statistics to disk.

Decouples the computation of dataset statistics from the I/O layer.
Outputs CSV files and a Markdown summary.
"""

from __future__ import annotations

import logging
from pathlib import Path

from trinetra.analysis.summary import AnalysisResult

logger = logging.getLogger(__name__)


def export_reports(result: AnalysisResult, output_dir: Path) -> None:
    """Export the computed analysis results to the filesystem.

    Creates the required directories and writes CSVs and a Markdown summary.

    Args:
        result: The computed analysis result containing DataFrames and Series.
        output_dir: The directory where the reports should be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export DataFrames to CSV
    dataset_csv = output_dir / "dataset_statistics.csv"
    result.dataset_stats.to_frame(name="Value").to_csv(dataset_csv)
    logger.info(f"Wrote {dataset_csv}")

    recording_csv = output_dir / "recording_statistics.csv"
    result.recording_stats_df.to_csv(recording_csv, index=False)
    logger.info(f"Wrote {recording_csv}")

    timestamp_csv = output_dir / "timestamp_statistics.csv"
    result.timestamp_stats_df.to_csv(timestamp_csv, index=False)
    logger.info(f"Wrote {timestamp_csv}")

    sensor_csv = output_dir / "sensor_statistics.csv"
    # sensor_stats_df has a multi-index (Sensor, Channel)
    result.sensor_stats_df.to_csv(sensor_csv)
    logger.info(f"Wrote {sensor_csv}")

    # 2. Export Summary to Markdown
    summary_md = output_dir / "summary.md"
    _write_markdown_summary(result, summary_md)
    logger.info(f"Wrote {summary_md}")


def _write_markdown_summary(result: AnalysisResult, filepath: Path) -> None:
    """Generate a readable Markdown summary of the analysis results."""
    ds = result.dataset_stats

    lines = [
        "# Trinetra-AI EDA Summary",
        "",
        "## Dataset Overview",
        "",
        f"- **Total Recordings**: {ds['total_recordings']}",
        f"- **Total Splits**: {ds['total_splits']}",
        f"- **Total Frames**: {ds['total_frames']}",
        f"- **Total Duration**: {ds['total_duration']:.2f} s",
        f"- **Min Duration**: {ds['min_duration']:.2f} s",
        f"- **Max Duration**: {ds['max_duration']:.2f} s",
        f"- **Average Duration**: {ds['avg_duration']:.2f} s",
        f"- **Average Sampling Frequency**: {ds['avg_sampling_frequency']:.2f} Hz",
        "",
        "## Splits",
        "",
    ]

    for split_name, count in result.recordings_per_split.items():
        lines.append(f"- **{split_name}**: {count} recordings")

    lines.extend(
        [
            "",
            "## Data Quality Flags",
            "",
        ]
    )

    # Check for non-monotonic or duplicated timestamps
    ts_df = result.timestamp_stats_df
    if not ts_df.empty:
        non_monotonic = (~ts_df["is_monotonic"]).sum()
        with_duplicates = (ts_df["duplicate_count"] > 0).sum()
        with_missing = (ts_df["missing_count"] > 0).sum()

        lines.append(f"- Recordings with non-monotonic timestamps: {non_monotonic}")
        lines.append(f"- Recordings with duplicated timestamps: {with_duplicates}")
        lines.append(f"- Recordings with missing timestamps: {with_missing}")
    else:
        lines.append("- No timestamp data available.")

    lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
