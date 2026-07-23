"""Immutable raw frame model for one synchronized RoNIN sensor sample.

This module defines :class:`RoninRawFrame`, the dataset-specific value object
produced by :class:`~.hdf5_reader.RoninHDF5Reader`.

Design contract
---------------
* ``RoninRawFrame`` is **adapter-layer only** — it MUST NOT be imported by
  any component in ``trinetra.domain`` or ``trinetra.application``.
* It carries data in the **exact units and coordinate frames stored in the
  HDF5 file**.  No unit conversion, normalization, or calibration is applied.
* The :class:`~.canonical_mapper.RoninCanonicalMapper` (M1.5) is the only
  component responsible for converting ``RoninRawFrame`` into the domain type
  ``SensorRecord``.

Schema source
-------------
Verified by direct inspection of ``data.hdf5`` files in the RoNIN dataset.
The ``synced`` group contains the following datasets (all shape ``(N, *)``,
dtype ``float64``):

+-------------------+----------+---------------------------------------------+
| HDF5 key          | Shape    | Description                                 |
+===================+==========+=============================================+
| ``synced/time``   | ``(N,)`` | IMU system timestamp (seconds)              |
+-------------------+----------+---------------------------------------------+
| ``synced/acce``   | ``(N,3)``| Accelerometer (m/s²), axes x, y, z         |
+-------------------+----------+---------------------------------------------+
| ``synced/gyro``   | ``(N,3)``| Gyroscope (rad/s), axes x, y, z            |
+-------------------+----------+---------------------------------------------+
| ``synced/gyro_uncalib`` | ``(N,3)`` | Uncalibrated gyroscope (rad/s)       |
+-------------------+----------+---------------------------------------------+
| ``synced/linacce``| ``(N,3)``| Linear acceleration (m/s²), gravity removed |
+-------------------+----------+---------------------------------------------+
| ``synced/grav``   | ``(N,3)``| Gravity vector (m/s²), axes x, y, z        |
+-------------------+----------+---------------------------------------------+
| ``synced/game_rv``| ``(N,4)``| Game rotation vector quaternion x, y, z, w  |
+-------------------+----------+---------------------------------------------+
| ``synced/rv``     | ``(N,4)``| Full rotation vector quaternion x, y, z, w  |
+-------------------+----------+---------------------------------------------+
| ``synced/magnet`` | ``(N,3)``| Magnetometer (µT), axes x, y, z            |
+-------------------+----------+---------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Type aliases for array shapes
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


@dataclass(frozen=True)
class RoninRawFrame:
    """One synchronized sensor sample decoded verbatim from ``synced/`` in
    a RoNIN ``data.hdf5`` file.

    All fields map 1-to-1 to HDF5 datasets in the ``synced`` group.  No
    unit conversion, calibration, or normalization is applied.

    Attributes:
        frame_index: Zero-based position of this frame within the recording,
            **after** the ``start_frame`` offset from ``info.json`` has been
            applied.  Frame 0 is therefore the first *usable* frame.
        timestamp: IMU system time in seconds (``synced/time``).
        acce: Accelerometer reading in m/s², axes (x, y, z)
            (``synced/acce``).
        gyro: Calibrated gyroscope reading in rad/s, axes (x, y, z)
            (``synced/gyro``).
        gyro_uncalib: Uncalibrated gyroscope reading in rad/s, axes
            (x, y, z) (``synced/gyro_uncalib``).
        linacce: Linear acceleration in m/s² with gravity removed, axes
            (x, y, z) (``synced/linacce``).
        grav: Gravity vector in m/s², axes (x, y, z) (``synced/grav``).
        game_rv: Game rotation vector as quaternion (x, y, z, w) —
            fused from accelerometer and gyroscope only, no magnetometer
            (``synced/game_rv``).
        rv: Full rotation vector as quaternion (x, y, z, w) — fused from
            accelerometer, gyroscope, and magnetometer (``synced/rv``).
        magnet: Magnetometer reading in µT, axes (x, y, z)
            (``synced/magnet``).

    Notes:
        This dataclass is deliberately *frozen* (immutable).  No methods,
        no business logic, and no conversions live here.
        See :class:`~.canonical_mapper.RoninCanonicalMapper` (M1.5) for the
        component that translates this into the domain ``SensorRecord`` type.
    """

    frame_index: int

    # -- Time -----------------------------------------------------------------
    timestamp: float  # seconds

    # -- Motion sensors -------------------------------------------------------
    acce: Vec3  # m/s² (x, y, z)
    gyro: Vec3  # rad/s (x, y, z)
    gyro_uncalib: Vec3  # rad/s (x, y, z)  uncalibrated
    linacce: Vec3  # m/s² (x, y, z)  gravity removed
    grav: Vec3  # m/s² (x, y, z)  gravity vector

    # -- Orientation sensors --------------------------------------------------
    game_rv: Vec4  # quaternion (x, y, z, w)  accel + gyro fusion
    rv: Vec4  # quaternion (x, y, z, w)  full fusion (incl. magnet)

    # -- Magnetic field -------------------------------------------------------
    magnet: Vec3  # µT (x, y, z)
