"""Aggregator and exporter for dataset sensor quality metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trinetra.analysis.plot_utils import SENSOR_ORDER

logger = logging.getLogger(__name__)


def aggregate_quality_results(results: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Convert a list of recording quality dictionaries into standardized DataFrames.

    Args:
        results: A list of quality dictionaries returned by assess_recording_quality.

    Returns:
        A dictionary containing the following DataFrames:
        - 'missing_values'
        - 'invalid_values'
        - 'outliers'
        - 'sensor_quality' (overall scores)
    """
    if not results:
        logger.warning("No quality results provided for aggregation.")
        return {
            "missing_values": pd.DataFrame(),
            "invalid_values": pd.DataFrame(),
            "outliers": pd.DataFrame(),
            "sensor_quality": pd.DataFrame(),
        }

    missing_rows = []
    invalid_rows = []
    outlier_rows = []
    quality_rows = []

    for r in results:
        rec_id = r["recording_id"]
        total = r["total_frames"]

        # Missing values (long format per sensor)
        for s in SENSOR_ORDER:
            missing_rows.append(
                {
                    "Recording": rec_id,
                    "Sensor": s,
                    "Total Frames": total,
                    "Missing Samples": r["missing_samples"].get(s, 0),
                    "NaN Count": r["nan_count"].get(s, 0),
                    "Inf Count": r["inf_count"].get(s, 0),
                }
            )

        # Invalid values (long format per sensor)
        for s in SENSOR_ORDER:
            inv_row = {
                "Recording": rec_id,
                "Sensor": s,
                "NaNs": r["nan_count"].get(s, 0),
                "Infs": r["inf_count"].get(s, 0),
                "Constant Signals": r["constant_values"].get(s, 0),
                "Saturation": r["saturation_count"].get(s, 0),
                "Duplicate Timestamps": r["temporal_consistency"][
                    "duplicate_timestamps"
                ],  # Same across sensors for a recording
            }
            if s == "orientation":
                inv_row["Quaternion Norm Violations"] = r["invalid_quaternions"]
            else:
                inv_row["Quaternion Norm Violations"] = 0
            invalid_rows.append(inv_row)

        # Outliers (long format)
        for s in SENSOR_ORDER:
            count = r["outlier_counts"].get(s, 0)
            pct = (count / total * 100.0) if total > 0 else 0.0
            outlier_rows.append(
                {
                    "Sensor": s,
                    "Recording": rec_id,
                    "Outlier Count": count,
                    "Outlier Percentage": round(pct, 2),
                    "Detection Method": "IQR (1.5)",
                }
            )

        # Overall quality row
        scores = r["scores"]
        quality_rows.append(
            {
                "Recording": rec_id,
                "Completeness": scores["completeness_score"],
                "Validity": scores["validity_score"],
                "Consistency": scores["consistency_score"],
                "Overall": scores["overall_score"],
                "Total Frames": total,
                "Missing": sum(r["missing_samples"].values()),
                "Outliers": sum(r["outlier_counts"].values()),
                "Timestamp Issues": r["temporal_consistency"]["duplicate_timestamps"]
                + r["temporal_consistency"]["non_monotonic"],
                "Invalid Quaternions": r["invalid_quaternions"],
                "NaNs": sum(r["nan_count"].values()),
            }
        )

    return {
        "missing_values": pd.DataFrame(missing_rows),
        "invalid_values": pd.DataFrame(invalid_rows),
        "outliers": pd.DataFrame(outlier_rows),
        "sensor_quality": pd.DataFrame(quality_rows),
    }


def export_quality_reports(dfs: dict[str, pd.DataFrame], output_dir: Path | str) -> None:
    """Write the quality DataFrames to CSV and generate a Markdown summary.

    Args:
        dfs: The dictionary of DataFrames from aggregate_quality_results.
        output_dir: The directory to save the reports.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write CSVs (dropping extra columns used only for the summary if needed)
    for name, df in dfs.items():
        if not df.empty:
            csv_path = out_path / f"{name}.csv"
            # Keep sensor_quality clean
            if name == "sensor_quality":
                csv_df = df[["Recording", "Completeness", "Validity", "Consistency", "Overall"]]
            else:
                csv_df = df
            csv_df.to_csv(csv_path, index=False)
            logger.info(f"Wrote {csv_path}")

    # Write Markdown Summary
    md_path = out_path / "quality_summary.md"

    sq_df = dfs.get("sensor_quality")
    if sq_df is not None and not sq_df.empty:

        # Determine score categories
        def categorize_score(s: float) -> str:
            if s >= 0.95:
                return "Excellent"
            if s >= 0.90:
                return "Good"
            if s >= 0.80:
                return "Acceptable"
            return "Review Required"

        sq_df["Category"] = sq_df["Overall"].apply(categorize_score)

        # Dataset-wide Summary Table
        total_recordings = len(sq_df)
        recs_with_nans = (sq_df["NaNs"] > 0).sum()
        recs_with_time_issues = (sq_df["Timestamp Issues"] > 0).sum()
        recs_with_quat_issues = (sq_df["Invalid Quaternions"] > 0).sum()
        avg_quality = sq_df["Overall"].mean()
        min_quality = sq_df["Overall"].min()
        max_quality = sq_df["Overall"].max()
        recs_with_high_outliers = (
            (sq_df["Outliers"] / sq_df["Total Frames"] > 0.01).sum() if total_recordings > 0 else 0
        )

        dataset_summary = pd.DataFrame(
            [
                {"Metric": "Total recordings", "Value": total_recordings},
                {"Metric": "Recordings with NaNs", "Value": recs_with_nans},
                {"Metric": "Recordings with timestamp issues", "Value": recs_with_time_issues},
                {"Metric": "Recordings with quaternion violations", "Value": recs_with_quat_issues},
                {"Metric": "Average quality score", "Value": f"{avg_quality:.4f}"},
                {"Metric": "Lowest quality score", "Value": f"{min_quality:.4f}"},
                {"Metric": "Highest quality score", "Value": f"{max_quality:.4f}"},
            ]
        )

        # Create the concise summary table requested by user
        summary_df = sq_df[
            ["Recording", "Missing", "Outliers", "Timestamp Issues", "Overall", "Category"]
        ].copy()
        summary_df.columns = [
            "Recording",
            "Missing",
            "Outliers",
            "Timestamp Issues",
            "Quality Score",
            "Category",
        ]

        md_content = [
            "# Sensor Quality Assessment Summary\n",
            "This report summarizes the structural, statistical, temporal, and signal consistency metrics over canonical SensorRecord streams.\n",
            "## Dataset Summary\n",
            dataset_summary.to_markdown(index=False),
            "\n",
            "## Recording-Level Quality Summary\n",
            summary_df.to_markdown(index=False),
            "\n",
            "## Key Findings\n",
            f"- {'No' if recs_with_nans == 0 else recs_with_nans} NaN values were detected across recordings.",
            f"- {'Timestamp monotonicity was preserved across all recordings.' if recs_with_time_issues == 0 else f'{recs_with_time_issues} recordings exhibited timestamp issues.'}",
            f"- {recs_with_high_outliers} recordings exhibited elevated outlier percentages (>1%).",
            f"- {'Quaternion normalization remained within tolerance.' if recs_with_quat_issues == 0 else f'{recs_with_quat_issues} recordings had quaternion norm violations.'}",
            f"- Average dataset quality score: {avg_quality:.2f}.\n",
        ]

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in md_content)

        logger.info(f"Wrote {md_path}")
    else:
        logger.warning("No sensor_quality data to write to Markdown.")
