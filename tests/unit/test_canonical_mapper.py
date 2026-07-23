"""Unit tests for RoninCanonicalMapper.

All tests use synthetic :class:`RoninRawFrame` instances constructed directly
in memory.  No HDF5 files, no real RoNIN dataset, and no filesystem access.
"""

from __future__ import annotations

import types

import pytest

from trinetra.adapters.datasets.ronin.canonical_mapper import RoninCanonicalMapper
from trinetra.adapters.datasets.ronin.raw_frames import RoninRawFrame
from trinetra.domain.interfaces.sensor_record import SensorRecord
from trinetra.shared.exceptions import DatasetError

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

FRAME_A = RoninRawFrame(
    frame_index=0,
    timestamp=100.5,
    acce=(0.10, 0.20, 9.81),
    gyro=(0.01, 0.02, 0.03),
    gyro_uncalib=(0.011, 0.021, 0.031),
    linacce=(0.10, 0.20, 0.00),
    grav=(0.00, 0.00, 9.81),
    game_rv=(0.00, 0.00, 0.00, 1.00),
    rv=(0.00, 0.00, 0.00, 1.00),
    magnet=(10.0, 20.0, 30.0),
)

FRAME_B = RoninRawFrame(
    frame_index=1,
    timestamp=100.505,
    acce=(0.11, 0.21, 9.82),
    gyro=(0.02, 0.03, 0.04),
    gyro_uncalib=(0.022, 0.032, 0.042),
    linacce=(0.11, 0.21, 0.01),
    grav=(0.01, 0.01, 9.81),
    game_rv=(0.01, 0.02, 0.03, 0.99),
    rv=(0.01, 0.02, 0.03, 0.99),
    magnet=(11.0, 21.0, 31.0),
)

FRAME_C = RoninRawFrame(
    frame_index=2,
    timestamp=100.510,
    acce=(0.12, 0.22, 9.83),
    gyro=(0.03, 0.04, 0.05),
    gyro_uncalib=(0.033, 0.043, 0.053),
    linacce=(0.12, 0.22, 0.02),
    grav=(0.02, 0.02, 9.81),
    game_rv=(0.02, 0.03, 0.04, 0.98),
    rv=(0.02, 0.03, 0.04, 0.98),
    magnet=(12.0, 22.0, 32.0),
)


@pytest.fixture()
def mapper() -> RoninCanonicalMapper:
    return RoninCanonicalMapper()


# ---------------------------------------------------------------------------
# Single-frame mapping
# ---------------------------------------------------------------------------


class TestMapFrame:
    def test_returns_sensor_record(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert isinstance(record, SensorRecord)

    def test_frame_id_mapped_from_frame_index(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.frame_id == FRAME_A.frame_index

    def test_timestamp_preserved(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.timestamp == pytest.approx(FRAME_A.timestamp)

    def test_accelerometer_mapped_from_acce(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.accelerometer == pytest.approx(FRAME_A.acce)

    def test_gyroscope_mapped_from_gyro(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.gyroscope == pytest.approx(FRAME_A.gyro)

    def test_linear_acceleration_mapped_from_linacce(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.linear_acceleration == pytest.approx(FRAME_A.linacce)

    def test_gravity_mapped_from_grav(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.gravity == pytest.approx(FRAME_A.grav)

    def test_orientation_mapped_from_game_rv(self, mapper: RoninCanonicalMapper) -> None:
        """orientation uses game_rv (accel+gyro fusion, no magnetometer)."""
        record = mapper.map_frame(FRAME_A)
        assert record.orientation == pytest.approx(FRAME_A.game_rv)

    def test_orientation_is_not_rv(self, mapper: RoninCanonicalMapper) -> None:
        """rv (full fusion with magnetometer) must NOT be used for orientation."""
        # Build a frame where game_rv != rv to verify the correct field is used.
        frame = RoninRawFrame(
            frame_index=0,
            timestamp=1.0,
            acce=(0.0, 0.0, 9.81),
            gyro=(0.0, 0.0, 0.0),
            gyro_uncalib=(0.0, 0.0, 0.0),
            linacce=(0.0, 0.0, 0.0),
            grav=(0.0, 0.0, 9.81),
            game_rv=(0.1, 0.2, 0.3, 0.9),  # different from rv
            rv=(0.5, 0.6, 0.7, 0.8),  # must NOT appear in SensorRecord.orientation
            magnet=(0.0, 0.0, 0.0),
        )
        record = mapper.map_frame(frame)
        assert record.orientation == pytest.approx((0.1, 0.2, 0.3, 0.9))
        assert record.orientation != pytest.approx((0.5, 0.6, 0.7, 0.8))

    def test_magnetometer_mapped_from_magnet(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert record.magnetometer == pytest.approx(FRAME_A.magnet)

    def test_accelerometer_is_tuple_of_three(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert len(record.accelerometer) == 3

    def test_orientation_is_tuple_of_four(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert len(record.orientation) == 4

    def test_magnetometer_is_tuple_of_three(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert len(record.magnetometer) == 3

    def test_different_frames_produce_different_records(self, mapper: RoninCanonicalMapper) -> None:
        rec_a = mapper.map_frame(FRAME_A)
        rec_b = mapper.map_frame(FRAME_B)
        assert rec_a.timestamp != pytest.approx(rec_b.timestamp)
        assert rec_a.frame_id != rec_b.frame_id

    def test_sensor_record_is_frozen(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        with pytest.raises((AttributeError, TypeError)):
            record.timestamp = 0.0  # type: ignore[misc]

    def test_sensor_record_type_fields_are_floats(self, mapper: RoninCanonicalMapper) -> None:
        record = mapper.map_frame(FRAME_A)
        assert isinstance(record.timestamp, float)
        assert all(isinstance(v, float) for v in record.accelerometer)
        assert all(isinstance(v, float) for v in record.gyroscope)
        assert all(isinstance(v, float) for v in record.linear_acceleration)
        assert all(isinstance(v, float) for v in record.gravity)
        assert all(isinstance(v, float) for v in record.orientation)
        assert all(isinstance(v, float) for v in record.magnetometer)


# ---------------------------------------------------------------------------
# None input
# ---------------------------------------------------------------------------


class TestNoneInput:
    def test_none_frame_raises_dataset_error(self, mapper: RoninCanonicalMapper) -> None:
        with pytest.raises(DatasetError):
            mapper.map_frame(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Iterable mapping
# ---------------------------------------------------------------------------


class TestMapFrames:
    def test_map_frames_returns_iterator(self, mapper: RoninCanonicalMapper) -> None:
        result = mapper.map_frames([FRAME_A, FRAME_B, FRAME_C])
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_map_frames_is_generator(self, mapper: RoninCanonicalMapper) -> None:
        result = mapper.map_frames([FRAME_A])
        assert isinstance(result, types.GeneratorType)

    def test_map_frames_yields_sensor_records(self, mapper: RoninCanonicalMapper) -> None:
        records = list(mapper.map_frames([FRAME_A, FRAME_B, FRAME_C]))
        assert all(isinstance(r, SensorRecord) for r in records)

    def test_map_frames_count_matches_input(self, mapper: RoninCanonicalMapper) -> None:
        records = list(mapper.map_frames([FRAME_A, FRAME_B, FRAME_C]))
        assert len(records) == 3

    def test_map_frames_empty_input_yields_nothing(self, mapper: RoninCanonicalMapper) -> None:
        records = list(mapper.map_frames([]))
        assert records == []

    def test_map_frames_preserves_order(self, mapper: RoninCanonicalMapper) -> None:
        records = list(mapper.map_frames([FRAME_A, FRAME_B, FRAME_C]))
        assert records[0].frame_id == FRAME_A.frame_index
        assert records[1].frame_id == FRAME_B.frame_index
        assert records[2].frame_id == FRAME_C.frame_index

    def test_map_frames_preserves_timestamps(self, mapper: RoninCanonicalMapper) -> None:
        records = list(mapper.map_frames([FRAME_A, FRAME_B, FRAME_C]))
        assert records[0].timestamp == pytest.approx(FRAME_A.timestamp)
        assert records[1].timestamp == pytest.approx(FRAME_B.timestamp)
        assert records[2].timestamp == pytest.approx(FRAME_C.timestamp)

    def test_map_frames_accepts_generator_input(self, mapper: RoninCanonicalMapper) -> None:
        """Verify map_frames accepts a generator (as yielded by RoninHDF5Reader)."""

        def frame_generator():
            yield FRAME_A
            yield FRAME_B

        records = list(mapper.map_frames(frame_generator()))
        assert len(records) == 2

    def test_map_frames_is_lazy_does_not_materialise(self, mapper: RoninCanonicalMapper) -> None:
        """Verify generator laziness: consuming 1 element should not evaluate all."""
        call_count = 0

        def counting_generator():
            nonlocal call_count
            for frame in [FRAME_A, FRAME_B, FRAME_C]:
                call_count += 1
                yield frame

        gen = mapper.map_frames(counting_generator())
        assert call_count == 0  # nothing consumed yet
        next(gen)
        assert call_count == 1  # only one frame evaluated

    def test_map_frames_with_single_frame(self, mapper: RoninCanonicalMapper) -> None:
        records = list(mapper.map_frames([FRAME_A]))
        assert len(records) == 1
        assert records[0].timestamp == pytest.approx(FRAME_A.timestamp)


# ---------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------


class TestStatelessness:
    def test_mapper_produces_same_output_on_repeated_calls(
        self, mapper: RoninCanonicalMapper
    ) -> None:
        rec_1 = mapper.map_frame(FRAME_A)
        rec_2 = mapper.map_frame(FRAME_A)
        assert rec_1 == rec_2

    def test_single_mapper_instance_handles_multiple_recordings(
        self, mapper: RoninCanonicalMapper
    ) -> None:
        """The mapper should be reusable across logically distinct frame sequences."""
        records_first = list(mapper.map_frames([FRAME_A, FRAME_B]))
        records_second = list(mapper.map_frames([FRAME_C]))
        assert records_first[0].frame_id == 0
        assert records_second[0].frame_id == 2

    def test_mapper_has_no_instance_state(self) -> None:
        """Two separate mapper instances must produce identical results."""
        mapper_1 = RoninCanonicalMapper()
        mapper_2 = RoninCanonicalMapper()
        assert mapper_1.map_frame(FRAME_A) == mapper_2.map_frame(FRAME_A)
