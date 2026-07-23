"""Unit tests for RoninHDF5Reader and RoninRawFrame.

All tests create synthetic HDF5 files in ``tmp_path``.
The real RoNIN dataset is NOT required.

Synthetic schema mirrors the real ``synced`` group exactly:

    synced/time        (N,)   float64
    synced/acce        (N,3)  float64
    synced/gyro        (N,3)  float64
    synced/gyro_uncalib (N,3) float64
    synced/linacce     (N,3)  float64
    synced/grav        (N,3)  float64
    synced/game_rv     (N,4)  float64
    synced/rv          (N,4)  float64
    synced/magnet      (N,3)  float64
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from trinetra.adapters.datasets.ronin.hdf5_reader import RoninHDF5Reader
from trinetra.adapters.datasets.ronin.metadata_models import (
    ImuCalibration,
    OrientationCalibration,
    OrientationErrors,
    RoninRecordingMetadata,
    TimeSynchronization,
)
from trinetra.adapters.datasets.ronin.models import Recording
from trinetra.adapters.datasets.ronin.raw_frames import RoninRawFrame
from trinetra.shared.exceptions import DatasetError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FRAMES = 20  # small synthetic recording length
START_FRAME = 5  # metadata.start_frame value used in tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recording(tmp_path: Path, *, create_hdf5: bool = True) -> Recording:
    """Create a minimal Recording pointing at a synthetic directory."""
    rec_dir = tmp_path / "a000_1"
    rec_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = rec_dir / "data.hdf5"
    info_path = rec_dir / "info.json"
    info_path.touch()
    if create_hdf5:
        hdf5_path.touch()  # real content written separately
    return Recording(
        recording_id="a000_1",
        split="train_dataset_1",
        recording_path=rec_dir,
        hdf5_path=hdf5_path,
        info_path=info_path,
    )


def _write_synced_group(
    hdf5_path: Path,
    n_frames: int = N_FRAMES,
    *,
    missing_datasets: list[str] | None = None,
    bad_length_dataset: str | None = None,
    bad_length: int = 5,
    omit_synced_group: bool = False,
) -> None:
    """Write a synthetic HDF5 file whose ``synced`` group mirrors the real schema."""
    rng = np.random.default_rng(seed=42)

    datasets = {
        "time": rng.uniform(0.0, 100.0, (n_frames,)),
        "acce": rng.standard_normal((n_frames, 3)),
        "gyro": rng.standard_normal((n_frames, 3)),
        "gyro_uncalib": rng.standard_normal((n_frames, 3)),
        "linacce": rng.standard_normal((n_frames, 3)),
        "grav": rng.standard_normal((n_frames, 3)),
        "game_rv": rng.standard_normal((n_frames, 4)),
        "rv": rng.standard_normal((n_frames, 4)),
        "magnet": rng.standard_normal((n_frames, 3)),
    }

    # Sort timestamps so they're monotonically increasing
    datasets["time"] = np.sort(datasets["time"])

    with h5py.File(hdf5_path, "w") as f:
        if omit_synced_group:
            return  # write an empty HDF5 (no synced group)

        synced = f.create_group("synced")
        for name, data in datasets.items():
            if missing_datasets and name in missing_datasets:
                continue
            if bad_length_dataset and name == bad_length_dataset:
                synced.create_dataset(name, data=data[:bad_length])
            else:
                synced.create_dataset(name, data=data)


def _make_metadata(start_frame: int = START_FRAME) -> RoninRecordingMetadata:
    """Build a minimal RoninRecordingMetadata for tests."""
    return RoninRecordingMetadata(
        recording_id="a000_1",
        split="train_dataset_1",
        length=100.0,
        date="01/08/19",
        device="asus5",
        recording_type="annotated",
        start_frame=start_frame,
        calibration=ImuCalibration(
            imu_init_gyro_bias=(0.0, 0.0, 0.0),
            imu_end_gyro_bias=(0.0, 0.0, 0.0),
            imu_acce_bias=(0.0, 0.0, 0.0),
            imu_acce_scale=(1.0, 1.0, 1.0),
        ),
        time_sync=TimeSynchronization(
            imu_reference_time=0.0,
            tango_reference_time=0.0,
            imu_time_offset=0.0,
        ),
        orientation=OrientationCalibration(
            align_tango_to_body=(0.0, 0.0, 0.0, 1.0),
            start_calibration=(0.0, 1.0, 0.0, 0.0),
            end_calibration=(0.0, 1.0, 0.0, 0.0),
        ),
        errors=OrientationErrors(
            gyro_integration_error=0.0,
            grv_ori_error=0.0,
            ekf_ori_error=0.0,
        ),
    )


@pytest.fixture()
def reader() -> RoninHDF5Reader:
    return RoninHDF5Reader()


@pytest.fixture()
def valid_recording_with_hdf5(tmp_path: Path) -> tuple[Recording, Path]:
    """Recording + a fully populated synthetic HDF5 file."""
    recording = _make_recording(tmp_path)
    _write_synced_group(recording.hdf5_path)
    return recording, recording.hdf5_path


# ---------------------------------------------------------------------------
# RoninRawFrame model tests
# ---------------------------------------------------------------------------


class TestRoninRawFrame:
    def test_construction(self) -> None:
        frame = RoninRawFrame(
            frame_index=0,
            timestamp=1.0,
            acce=(0.1, 0.2, 0.3),
            gyro=(0.01, 0.02, 0.03),
            gyro_uncalib=(0.011, 0.021, 0.031),
            linacce=(0.05, 0.06, 0.07),
            grav=(0.0, 0.0, 9.81),
            game_rv=(0.0, 0.0, 0.0, 1.0),
            rv=(0.0, 0.0, 0.0, 1.0),
            magnet=(10.0, 20.0, 30.0),
        )
        assert frame.timestamp == pytest.approx(1.0)
        assert frame.acce == pytest.approx((0.1, 0.2, 0.3))

    def test_immutability(self) -> None:
        frame = RoninRawFrame(
            frame_index=0,
            timestamp=1.0,
            acce=(0.1, 0.2, 0.3),
            gyro=(0.01, 0.02, 0.03),
            gyro_uncalib=(0.011, 0.021, 0.031),
            linacce=(0.05, 0.06, 0.07),
            grav=(0.0, 0.0, 9.81),
            game_rv=(0.0, 0.0, 0.0, 1.0),
            rv=(0.0, 0.0, 0.0, 1.0),
            magnet=(10.0, 20.0, 30.0),
        )
        with pytest.raises((AttributeError, TypeError)):
            frame.timestamp = 99.0  # type: ignore[misc]

    def test_acce_is_tuple_of_three(self) -> None:
        frame = RoninRawFrame(
            frame_index=0,
            timestamp=0.0,
            acce=(1.0, 2.0, 3.0),
            gyro=(0.0, 0.0, 0.0),
            gyro_uncalib=(0.0, 0.0, 0.0),
            linacce=(0.0, 0.0, 0.0),
            grav=(0.0, 0.0, 9.81),
            game_rv=(0.0, 0.0, 0.0, 1.0),
            rv=(0.0, 0.0, 0.0, 1.0),
            magnet=(0.0, 0.0, 0.0),
        )
        assert isinstance(frame.acce, tuple)
        assert len(frame.acce) == 3

    def test_game_rv_is_tuple_of_four(self) -> None:
        frame = RoninRawFrame(
            frame_index=0,
            timestamp=0.0,
            acce=(0.0, 0.0, 0.0),
            gyro=(0.0, 0.0, 0.0),
            gyro_uncalib=(0.0, 0.0, 0.0),
            linacce=(0.0, 0.0, 0.0),
            grav=(0.0, 0.0, 9.81),
            game_rv=(0.1, 0.2, 0.3, 0.9),
            rv=(0.0, 0.0, 0.0, 1.0),
            magnet=(0.0, 0.0, 0.0),
        )
        assert isinstance(frame.game_rv, tuple)
        assert len(frame.game_rv) == 4


# ---------------------------------------------------------------------------
# Valid HDF5
# ---------------------------------------------------------------------------


class TestValidHdf5:
    def test_read_returns_generator(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        import types

        result = reader.read(recording)
        assert isinstance(result, types.GeneratorType)

    def test_read_yields_ronin_raw_frames(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert all(isinstance(f, RoninRawFrame) for f in frames)

    def test_read_without_metadata_yields_all_frames(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert len(frames) == N_FRAMES

    def test_read_with_metadata_applies_start_frame(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        metadata = _make_metadata(start_frame=START_FRAME)
        frames = list(reader.read(recording, metadata))
        assert len(frames) == N_FRAMES - START_FRAME

    def test_frame_index_starts_at_zero(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert frames[0].frame_index == 0

    def test_frame_indices_are_sequential(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        for i, frame in enumerate(frames):
            assert frame.frame_index == i

    def test_timestamp_is_float(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert isinstance(frames[0].timestamp, float)

    def test_acce_has_correct_shape(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert len(frames[0].acce) == 3

    def test_gyro_has_correct_shape(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert len(frames[0].gyro) == 3

    def test_game_rv_has_four_components(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert len(frames[0].game_rv) == 4

    def test_rv_has_four_components(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert len(frames[0].rv) == 4

    def test_magnet_has_three_components(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames = list(reader.read(recording))
        assert len(frames[0].magnet) == 3

    def test_all_frame_fields_are_floats(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frame = next(reader.read(recording))
        assert isinstance(frame.timestamp, float)
        assert all(isinstance(v, float) for v in frame.acce)
        assert all(isinstance(v, float) for v in frame.gyro)
        assert all(isinstance(v, float) for v in frame.gyro_uncalib)
        assert all(isinstance(v, float) for v in frame.linacce)
        assert all(isinstance(v, float) for v in frame.grav)
        assert all(isinstance(v, float) for v in frame.game_rv)
        assert all(isinstance(v, float) for v in frame.rv)
        assert all(isinstance(v, float) for v in frame.magnet)

    def test_reader_is_reusable(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        frames_a = list(reader.read(recording))
        frames_b = list(reader.read(recording))
        assert len(frames_a) == len(frames_b)

    def test_start_frame_zero_yields_all_frames(
        self,
        reader: RoninHDF5Reader,
        valid_recording_with_hdf5: tuple[Recording, Path],
    ) -> None:
        recording, _ = valid_recording_with_hdf5
        metadata = _make_metadata(start_frame=0)
        frames = list(reader.read(recording, metadata))
        assert len(frames) == N_FRAMES


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


class TestMissingFile:
    def test_missing_hdf5_file_raises_dataset_error(
        self, reader: RoninHDF5Reader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path, create_hdf5=False)
        with pytest.raises(DatasetError, match="not found"):
            list(reader.read(recording))

    def test_error_contains_recording_id(self, reader: RoninHDF5Reader, tmp_path: Path) -> None:
        recording = _make_recording(tmp_path, create_hdf5=False)
        with pytest.raises(DatasetError, match="a000_1"):
            list(reader.read(recording))


# ---------------------------------------------------------------------------
# Missing synced group
# ---------------------------------------------------------------------------


class TestMissingSyncedGroup:
    def test_missing_synced_group_raises_dataset_error(
        self, reader: RoninHDF5Reader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path)
        _write_synced_group(recording.hdf5_path, omit_synced_group=True)
        with pytest.raises(DatasetError, match="synced"):
            list(reader.read(recording))


# ---------------------------------------------------------------------------
# Missing required datasets
# ---------------------------------------------------------------------------


REQUIRED = [
    "time",
    "acce",
    "gyro",
    "gyro_uncalib",
    "linacce",
    "grav",
    "game_rv",
    "rv",
    "magnet",
]


class TestMissingRequiredDatasets:
    @pytest.mark.parametrize("missing_key", REQUIRED)
    def test_missing_dataset_raises_dataset_error(
        self,
        reader: RoninHDF5Reader,
        tmp_path: Path,
        missing_key: str,
    ) -> None:
        recording = _make_recording(tmp_path / missing_key)
        _write_synced_group(recording.hdf5_path, missing_datasets=[missing_key])
        with pytest.raises(DatasetError, match=missing_key):
            list(reader.read(recording))


# ---------------------------------------------------------------------------
# Inconsistent dataset lengths
# ---------------------------------------------------------------------------


class TestInconsistentLengths:
    def test_inconsistent_lengths_raises_dataset_error(
        self, reader: RoninHDF5Reader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path)
        _write_synced_group(
            recording.hdf5_path,
            bad_length_dataset="acce",
            bad_length=3,
        )
        with pytest.raises(DatasetError, match="Inconsistent"):
            list(reader.read(recording))

    def test_error_mentions_inconsistent_key(self, reader: RoninHDF5Reader, tmp_path: Path) -> None:
        recording = _make_recording(tmp_path)
        _write_synced_group(
            recording.hdf5_path,
            bad_length_dataset="gyro",
            bad_length=2,
        )
        with pytest.raises(DatasetError, match="gyro"):
            list(reader.read(recording))


# ---------------------------------------------------------------------------
# start_frame out of range
# ---------------------------------------------------------------------------


class TestStartFrameValidation:
    def test_start_frame_equal_to_total_frames_raises(
        self, reader: RoninHDF5Reader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path)
        _write_synced_group(recording.hdf5_path, n_frames=N_FRAMES)
        metadata = _make_metadata(start_frame=N_FRAMES)  # out of range
        with pytest.raises(DatasetError, match="start_frame"):
            list(reader.read(recording, metadata))

    def test_start_frame_beyond_total_frames_raises(
        self, reader: RoninHDF5Reader, tmp_path: Path
    ) -> None:
        recording = _make_recording(tmp_path)
        _write_synced_group(recording.hdf5_path, n_frames=N_FRAMES)
        metadata = _make_metadata(start_frame=N_FRAMES + 100)
        with pytest.raises(DatasetError, match="start_frame"):
            list(reader.read(recording, metadata))
