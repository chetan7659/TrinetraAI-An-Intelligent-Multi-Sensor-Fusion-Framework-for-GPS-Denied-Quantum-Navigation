"""Orchestrator for Exploratory Data Analysis (EDA).

Coordinates a single-pass streaming analysis of the dataset, producing
comprehensive statistical dataframes and metrics without performing any
filesystem serialization.
"""

from __future__ import annotations

import logging

import pandas as pd

from trinetra.analysis.dataset_statistics import (
    compute_dataset_statistics,
    compute_recordings_per_split,
)
from trinetra.analysis.recording_statistics import RecordingStatsAggregator
from trinetra.analysis.sensor_statistics import SensorStatsAggregator
from trinetra.analysis.timestamp_statistics import TimestampStatsAggregator
from trinetra.application.dataset.split_loader import SplitLoader

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Container for the computed statistics DataFrames and Series."""

    def __init__(
        self,
        dataset_stats: pd.Series,
        recordings_per_split: pd.Series,
        recording_stats_df: pd.DataFrame,
        timestamp_stats_df: pd.DataFrame,
        sensor_stats_df: pd.DataFrame,
    ) -> None:
        self.dataset_stats = dataset_stats
        self.recordings_per_split = recordings_per_split
        self.recording_stats_df = recording_stats_df
        self.timestamp_stats_df = timestamp_stats_df
        self.sensor_stats_df = sensor_stats_df


def generate_statistics(loader: SplitLoader, splits: list[str]) -> AnalysisResult:
    """Generate comprehensive dataset statistics via single-pass streaming.

    Args:
        loader: The SplitLoader orchestrating the dataset traversal.
        splits: The list of dataset partitions (e.g., ["train", "seen", "unseen"])
            to include in the analysis.

    Returns:
        An AnalysisResult object containing Pandas DataFrames and Series
        representing the computed statistics.
    """
    recording_stats_list = []
    timestamp_stats_list = []

    # SensorStatsAggregator accumulates globally across the entire dataset
    global_sensor_stats = SensorStatsAggregator()

    for split in splits:
        logger.info(f"Analyzing split: {split}")
        # Iterating at the recording level allows us to instantiate
        # per-recording aggregators. We use loader.iter_recordings to get
        # the recordings, then the loader's underlying iterator to get frames.

        try:
            recordings = list(loader.iter_recordings(split))
        except Exception as e:
            logger.error(f"Failed to list recordings for split {split}: {e}")
            continue

        for recording in recordings:
            recording_id = getattr(recording, "id", "unknown")
            logger.debug(f"Processing recording: {recording_id}")

            rec_agg = RecordingStatsAggregator(recording_id, split)
            ts_agg = TimestampStatsAggregator(recording_id, split)

            # Single pass over the recording's sensor records
            try:
                for record in loader._recording_iterator.iter_recording(recording):
                    rec_agg.update(record)
                    ts_agg.update(record)
                    global_sensor_stats.update(record)
            except Exception as e:
                logger.error(f"Error processing recording {recording_id}: {e}")

            recording_stats_list.append(rec_agg.finalize())
            timestamp_stats_list.append(ts_agg.finalize())

    # Compile the final dataset-level views
    dataset_stats = compute_dataset_statistics(recording_stats_list)
    recordings_per_split = compute_recordings_per_split(recording_stats_list)

    recording_stats_df = pd.DataFrame(recording_stats_list)
    timestamp_stats_df = pd.DataFrame(timestamp_stats_list)
    sensor_stats_df = global_sensor_stats.finalize()

    return AnalysisResult(
        dataset_stats=dataset_stats,
        recordings_per_split=recordings_per_split,
        recording_stats_df=recording_stats_df,
        timestamp_stats_df=timestamp_stats_df,
        sensor_stats_df=sensor_stats_df,
    )
