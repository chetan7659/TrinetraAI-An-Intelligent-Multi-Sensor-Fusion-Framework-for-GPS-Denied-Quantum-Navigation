"""Unit tests for EDA summary orchestration."""

from __future__ import annotations

import typing
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest

from trinetra.analysis.summary import AnalysisResult, generate_statistics
from trinetra.application.dataset.split_loader import SplitLoader
from trinetra.domain.interfaces.sensor_record import SensorRecord

# ---------------------------------------------------------------------------
# Synthetic Data Factory
# ---------------------------------------------------------------------------


class StubRecording:
    def __init__(self, rec_id: str) -> None:
        self.id = rec_id


REC_1 = StubRecording("rec_1")
REC_2 = StubRecording("rec_2")


def _make_record(frame_id: int, timestamp: float) -> SensorRecord:
    return SensorRecord(
        frame_id=frame_id,
        timestamp=timestamp,
        accelerometer=(0.0, 0.0, 0.0),
        gyroscope=(0.0, 0.0, 0.0),
        linear_acceleration=(0.0, 0.0, 0.0),
        gravity=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        magnetometer=(0.0, 0.0, 0.0),
    )


RECORDING_MAP = {
    REC_1: [_make_record(0, 0.0), _make_record(1, 0.5)],
    REC_2: [_make_record(2, 0.0), _make_record(3, 1.0), _make_record(4, 2.0)],
}


# ---------------------------------------------------------------------------
# Collaborator stubs
# ---------------------------------------------------------------------------


class StubAdapter:
    SPLITS: typing.ClassVar[dict[str, list[Any]]] = {
        "train": [REC_1, REC_2],
        "test": [],
    }

    def list_recordings(self, split: str) -> list[Any]:
        if split not in self.SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return self.SPLITS[split]


class StubRecordingIterator:
    def iter_recording(self, recording: Any) -> Iterator[SensorRecord]:
        yield from RECORDING_MAP[recording]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def loader() -> SplitLoader:
    return SplitLoader(StubAdapter(), StubRecordingIterator())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_statistics_success(loader: SplitLoader) -> None:
    result = generate_statistics(loader, ["train", "test"])

    assert isinstance(result, AnalysisResult)

    # Check dataset stats
    assert result.dataset_stats["total_recordings"] == 2
    assert result.dataset_stats["total_splits"] == 1  # Only "train" had recordings
    assert result.dataset_stats["total_frames"] == 5

    # Check recordings per split
    assert result.recordings_per_split["train"] == 2

    # Check recording stats df
    assert isinstance(result.recording_stats_df, pd.DataFrame)
    assert len(result.recording_stats_df) == 2

    # Check timestamp stats df
    assert isinstance(result.timestamp_stats_df, pd.DataFrame)
    assert len(result.timestamp_stats_df) == 2

    # Check sensor stats df
    assert isinstance(result.sensor_stats_df, pd.DataFrame)
    # At least 19 channels
    assert len(result.sensor_stats_df) == 19
    # We pushed 5 frames in total across all recordings
    assert (result.sensor_stats_df["count"] == 5).all()


def test_generate_statistics_missing_split(loader: SplitLoader) -> None:
    # Error should be caught and logged, remaining splits processed
    result = generate_statistics(loader, ["unknown", "train"])

    assert result.dataset_stats["total_recordings"] == 2
    assert len(result.recording_stats_df) == 2
