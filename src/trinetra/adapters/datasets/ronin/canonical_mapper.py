"""Canonical Mapper for the RoNIN dataset.

This module provides :class:`RoninCanonicalMapper`, which translates
dataset-specific :class:`~.raw_frames.RoninRawFrame` objects into the
project-wide canonical domain type :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`.

Design contract (ADR-0002)
--------------------------
* This is the **only** component in the RoNIN adapter package that imports
  from ``trinetra.domain``.
* It is **stateless** — no instance state, no side effects, no I/O.
* It is **pure** — identical input always produces identical output.
* It does **not** open files, parse JSON, call h5py, or apply calibration.
* All unit and coordinate-frame conventions are preserved verbatim from the
  :class:`~.raw_frames.RoninRawFrame`; no normalization, rescaling, or
  resampling is performed.

Orientation convention
----------------------
``SensorRecord.orientation`` is populated from ``RoninRawFrame.game_rv``
(the game rotation vector — fused from accelerometer and gyroscope only,
independent of the magnetometer).  This is the orientation signal used in the
RoNIN training labels and is the most portable choice across recording sessions
where magnetometer reliability varies.

``RoninRawFrame.rv`` (the full rotation vector, which fuses the magnetometer)
is intentionally omitted from ``SensorRecord`` to keep the canonical type
free of magnetometer-dependent orientation estimates.  If a downstream
component requires ``rv``, it must hold a reference to the raw frame.

Future datasets
---------------
Each new dataset implements its own mapper following this exact pattern::

    class TlioCanonicalMapper:
        def map_frame(self, frame: TlioRawFrame) -> SensorRecord: ...
        def map_frames(self, frames: Iterable[TlioRawFrame]) -> Iterator[SensorRecord]: ...

The Application layer and Domain layer always receive ``SensorRecord`` and
never need to know which mapper or dataset produced them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from trinetra.domain.interfaces.sensor_record import SensorRecord
from trinetra.shared.exceptions import DatasetError

from .raw_frames import RoninRawFrame


class RoninCanonicalMapper:
    """Translates :class:`~.raw_frames.RoninRawFrame` objects produced by
    :class:`~.hdf5_reader.RoninHDF5Reader` into the canonical
    :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord` domain type.

    This class is **stateless** — a single instance may safely be reused
    across multiple recordings, splits, and datasets.

    Example::

        from trinetra.adapters.datasets.ronin import (
            RoninAdapter,
            RoninMetadataLoader,
            RoninHDF5Reader,
            RoninCanonicalMapper,
        )

        adapter   = RoninAdapter()
        recording = adapter.get_recording("a000_1")

        metadata  = RoninMetadataLoader().load(recording)
        frames    = RoninHDF5Reader().read(recording, metadata)

        mapper    = RoninCanonicalMapper()
        records   = mapper.map_frames(frames)

        for record in records:
            print(record.timestamp, record.accelerometer)
    """

    def map_frame(self, frame: RoninRawFrame) -> SensorRecord:
        """Translate one :class:`~.raw_frames.RoninRawFrame` into one
        :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`.

        Mapping rules
        -------------
        +---------------------------+------------------------------+
        | ``RoninRawFrame`` field   | ``SensorRecord`` field       |
        +===========================+==============================+
        | ``frame_index``           | ``frame_id``                 |
        +---------------------------+------------------------------+
        | ``timestamp``             | ``timestamp``                |
        +---------------------------+------------------------------+
        | ``acce``                  | ``accelerometer``            |
        +---------------------------+------------------------------+
        | ``gyro``                  | ``gyroscope``                |
        +---------------------------+------------------------------+
        | ``linacce``               | ``linear_acceleration``      |
        +---------------------------+------------------------------+
        | ``grav``                  | ``gravity``                  |
        +---------------------------+------------------------------+
        | ``game_rv``               | ``orientation``              |
        +---------------------------+------------------------------+
        | ``magnet``                | ``magnetometer``             |
        +---------------------------+------------------------------+

        Args:
            frame: A :class:`~.raw_frames.RoninRawFrame` as yielded by
                :class:`~.hdf5_reader.RoninHDF5Reader`.

        Returns:
            An immutable :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`.

        Raises:
            DatasetError: If ``frame`` is ``None``.
        """
        if frame is None:
            raise DatasetError("[RoninCanonicalMapper] Received None instead of a RoninRawFrame.")

        return SensorRecord(
            frame_id=frame.frame_index,
            timestamp=frame.timestamp,
            accelerometer=frame.acce,
            gyroscope=frame.gyro,
            linear_acceleration=frame.linacce,
            gravity=frame.grav,
            orientation=frame.game_rv,
            magnetometer=frame.magnet,
        )

    def map_frames(self, frames: Iterable[RoninRawFrame]) -> Iterator[SensorRecord]:
        """Lazily translate an iterable of :class:`~.raw_frames.RoninRawFrame`
        objects into an iterator of :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`.

        This method is a generator — no frames are materialised into a list.
        It is safe to pipe the output of :meth:`~.hdf5_reader.RoninHDF5Reader.read`
        directly into this method.

        Args:
            frames: Any iterable of :class:`~.raw_frames.RoninRawFrame`, typically
                the generator returned by :meth:`~.hdf5_reader.RoninHDF5Reader.read`.

        Yields:
            One :class:`~trinetra.domain.interfaces.sensor_record.SensorRecord`
            per input frame, in the same order.

        Raises:
            DatasetError: If any frame in the iterable is ``None``.
        """
        for frame in frames:
            yield self.map_frame(frame)
