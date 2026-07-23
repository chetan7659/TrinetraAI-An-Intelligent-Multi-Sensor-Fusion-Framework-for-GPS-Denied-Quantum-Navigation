"""Unit tests for quality summary aggregation and export."""

from __future__ import annotations

from trinetra.analysis.quality_summary import aggregate_quality_results


def test_aggregate_quality_results_empty() -> None:
    dfs = aggregate_quality_results([])
    assert len(dfs) == 4
    for df in dfs.values():
        assert df.empty


def test_aggregate_quality_results() -> None:
    results = [
        {
            "recording_id": "rec_01",
            "total_frames": 100,
            "missing_samples": {"accelerometer": 5, "gyroscope": 0},
            "nan_count": {"accelerometer": 1, "gyroscope": 0},
            "inf_count": {"accelerometer": 0, "gyroscope": 0},
            "zero_vectors": {"accelerometer": 0, "gyroscope": 0},
            "constant_values": {"accelerometer": 0, "gyroscope": 0},
            "saturation_count": {"accelerometer": 2, "gyroscope": 0},
            "outlier_counts": {"accelerometer": 10, "gyroscope": 2},
            "invalid_quaternions": 1,
            "temporal_consistency": {
                "duplicate_timestamps": 1,
                "non_monotonic": 0,
            },
            "scores": {
                "completeness_score": 0.95,
                "consistency_score": 0.99,
                "validity_score": 0.90,
                "overall_score": 0.94,
            },
        }
    ]

    dfs = aggregate_quality_results(results)

    assert len(dfs) == 4

    # Check missing values df (long format)
    mv = dfs["missing_values"]
    assert len(mv) == 6  # 6 sensors
    accel_mv = mv[(mv["Recording"] == "rec_01") & (mv["Sensor"] == "accelerometer")]
    assert accel_mv.iloc[0]["Missing Samples"] == 5

    # Check invalid values df (long format)
    iv = dfs["invalid_values"]
    assert len(iv) == 6
    accel_iv = iv[(iv["Recording"] == "rec_01") & (iv["Sensor"] == "accelerometer")]
    assert accel_iv.iloc[0]["Saturation"] == 2

    # Check outliers df (long format)
    out = dfs["outliers"]
    assert len(out) == 6
    accel_out = out[(out["Recording"] == "rec_01") & (out["Sensor"] == "accelerometer")]
    assert accel_out.iloc[0]["Outlier Count"] == 10
    assert accel_out.iloc[0]["Outlier Percentage"] == 10.0

    # Check quality df
    qual = dfs["sensor_quality"]
    assert qual.iloc[0]["Missing"] == 5
    assert qual.iloc[0]["Outliers"] == 12
    assert qual.iloc[0]["Timestamp Issues"] == 1
    assert qual.iloc[0]["Overall"] == 0.94
