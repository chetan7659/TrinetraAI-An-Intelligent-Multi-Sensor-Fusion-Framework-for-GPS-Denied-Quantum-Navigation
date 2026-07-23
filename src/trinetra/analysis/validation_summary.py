"""Aggregator, orchestrator, and report exporter for coordinate and orientation validation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trinetra.analysis.coordinate_validation import check_coordinate_consistency, validate_gravity
from trinetra.analysis.frame_validation import validate_sensor_frame_consistency
from trinetra.analysis.orientation_validation import (
    DEFAULT_MAX_ANGULAR_STEP_RAD,
    validate_quaternions,
)
from trinetra.domain.interfaces.sensor_record import SensorRecord

logger = logging.getLogger(__name__)


def validate_recording_geometry(
    recording_id: str, stream: Iterable[SensorRecord]
) -> dict[str, Any]:
    """Validate coordinate and orientation geometry of a single recording stream in a single pass.

    Args:
        recording_id: Unique recording identifier.
        stream: Stream of SensorRecord objects.

    Returns:
        Dictionary of validation metrics for the recording.
    """
    gravity_vecs = []
    accel_vecs = []
    linear_accel_vecs = []
    quats = []

    for record in stream:
        if record.gravity is not None:
            gravity_vecs.append(record.gravity)
        if record.accelerometer is not None:
            accel_vecs.append(record.accelerometer)
        if record.linear_acceleration is not None:
            linear_accel_vecs.append(record.linear_acceleration)
        if record.orientation is not None:
            quats.append(record.orientation)

    # Compute coordinate and gravity stats
    grav_stats = validate_gravity(gravity_vecs)
    coord_stats = check_coordinate_consistency(accel_vecs, gravity_vecs)

    # Compute orientation stats
    orient_stats = validate_quaternions(quats)

    # Compute frame consistency (residual check)
    frame_stats = validate_sensor_frame_consistency(accel_vecs, gravity_vecs, linear_accel_vecs)

    # Classify overall status (PASS, REVIEW, FAIL)
    # Define thresholds
    norm_mean = orient_stats.get("mean_norm", 1.0)
    norm_err = abs(norm_mean - 1.0) if len(quats) > 0 else 0.0
    gravity_bias = abs(grav_stats.get("bias", 0.0)) if len(gravity_vecs) > 0 else 0.0
    jumps = orient_stats.get("abnormal_jump_count", 0.0)

    # Rules
    if norm_err > 0.05 or gravity_bias > 1.0 or jumps > 20:
        status = "FAIL"
    elif (
        norm_err > 0.01
        or gravity_bias > 0.1
        or jumps > 0
        or coord_stats.get("axis_polarities_consistent", 1.0) == 0.0
    ):
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "recording_id": recording_id,
        "status": status,
        "gravity": grav_stats,
        "coordinate": coord_stats,
        "orientation": orient_stats,
        "frame": frame_stats,
    }


def aggregate_validation_results(results: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Aggregate recording validation dictionaries into Pandas DataFrames.

    Args:
        results: List of validation results from validate_recording_geometry.

    Returns:
        Dictionary containing coordinate_checks, orientation_checks, and frame_validation DataFrames.
    """
    if not results:
        return {
            "coordinate_checks": pd.DataFrame(),
            "orientation_checks": pd.DataFrame(),
            "frame_validation": pd.DataFrame(),
        }

    coord_rows = []
    orient_rows = []
    frame_rows = []

    for r in results:
        rec_id = r["recording_id"]
        status = r["status"]

        # Coordinate checks
        c = r["coordinate"]
        g = r["gravity"]
        coord_rows.append(
            {
                "Recording": rec_id,
                "Status": status,
                "Mean Gravity Mag": g["mean_magnitude"],
                "Gravity Bias": g["bias"],
                "Mean Accel Mag": c["accel_mean_magnitude"],
                "Gravity Axis": c["gravity_alignment_axis"],
                "Axis Consistency": c["axis_polarities_consistent"],
            }
        )

        # Orientation checks
        o = r["orientation"]
        orient_rows.append(
            {
                "Recording": rec_id,
                "Status": status,
                "Min Norm": o["min_norm"],
                "Max Norm": o["max_norm"],
                "Mean Norm": o["mean_norm"],
                "Sign Flips": o["sign_flip_count"],
                "Abnormal Jumps": o["abnormal_jump_count"],
                "Mean Angular Step": o["mean_angular_step_rad"],
                "Max Angular Step": o["max_angular_step_rad"],
            }
        )

        # Frame checks
        f = r["frame"]
        frame_rows.append(
            {
                "Recording": rec_id,
                "Status": status,
                "Mean Residual": f["mean_residual"],
                "RMS Residual": f["rms_residual"],
                "Max Residual": f["max_residual"],
                "P95 Residual": f["p95_residual"],
            }
        )

    return {
        "coordinate_checks": pd.DataFrame(coord_rows),
        "orientation_checks": pd.DataFrame(orient_rows),
        "frame_validation": pd.DataFrame(frame_rows),
    }


def export_validation_reports(dfs: dict[str, pd.DataFrame], output_dir: Path | str) -> None:
    """Export the validation DataFrames to CSV and write a Markdown report.

    Args:
        dfs: Aggregated validation DataFrames.
        output_dir: Directory path to save files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    for name, df in dfs.items():
        if not df.empty:
            csv_path = out_path / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Wrote {csv_path}")

    # Write Markdown Summary
    md_path = out_path / "validation_summary.md"

    coord_df = dfs.get("coordinate_checks")
    if coord_df is not None and not coord_df.empty:
        total_recordings = len(coord_df)
        status_counts = coord_df["Status"].value_counts().to_dict()
        pass_count = status_counts.get("PASS", 0)
        review_count = status_counts.get("REVIEW", 0)
        fail_count = status_counts.get("FAIL", 0)

        # Summary metrics
        avg_bias = coord_df["Gravity Bias"].abs().mean()

        dataset_summary = pd.DataFrame(
            [
                {"Metric": "Total recordings", "Value": total_recordings},
                {"Metric": "PASS status count", "Value": pass_count},
                {"Metric": "REVIEW status count", "Value": review_count},
                {"Metric": "FAIL status count", "Value": fail_count},
                {"Metric": "Average gravity magnitude bias", "Value": f"{avg_bias:.4f} m/s^2"},
            ]
        )

        # Validation rules definition
        rules_df = pd.DataFrame(
            [
                {
                    "Criterion": "Quaternion norm error",
                    "Warning (REVIEW)": "> 0.01",
                    "Failure (FAIL)": "> 0.05",
                },
                {
                    "Criterion": "Gravity magnitude bias",
                    "Warning (REVIEW)": "> 0.1 m/s^2",
                    "Failure (FAIL)": "> 1.0 m/s^2",
                },
                {
                    "Criterion": "Abnormal angular jumps",
                    "Warning (REVIEW)": "> 0",
                    "Failure (FAIL)": "> 20",
                },
                {
                    "Criterion": "Axis polarity flips",
                    "Warning (REVIEW)": "Detected",
                    "Failure (FAIL)": "-",
                },
            ]
        )

        # Recording level summary table
        rec_summary_df = coord_df[
            ["Recording", "Status", "Gravity Bias", "Axis Consistency"]
        ].copy()

        md_content = [
            "# Sensor Coordinate & Orientation Validation Summary\n",
            "This report summarizes the coordinate consistency, quaternion orientation continuity, and physical cross-sensor coherence checks over canonical SensorRecord streams.\n",
            "## Validation Criteria\n",
            f"The checking processes use standard thresholds (abnormal angular step defined as > {np.rad2deg(DEFAULT_MAX_ANGULAR_STEP_RAD):.1f}° or {DEFAULT_MAX_ANGULAR_STEP_RAD:.4f} rad):\n",
            rules_df.to_markdown(index=False),
            "\n",
            "## Dataset Summary\n",
            dataset_summary.to_markdown(index=False),
            "\n",
            "## Recording-Level Validation Status\n",
            rec_summary_df.to_markdown(index=False),
            "\n",
            "## Key Findings\n",
            f"- Out of {total_recordings} recordings: {pass_count} passed, {review_count} require review, and {fail_count} failed consistency checks.",
            f"- Average absolute gravity bias is {avg_bias:.4f} m/s^2.",
            f"- {'No recordings failed. The dataset is geometrically sound for navigation.' if fail_count == 0 else f'{fail_count} recordings show major geometric failures; preprocessing or exclusion is advised.'}",
            "\n",
        ]

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in md_content)

        logger.info(f"Wrote {md_path}")
    else:
        logger.warning("No coordinate checks data available for summary report.")
