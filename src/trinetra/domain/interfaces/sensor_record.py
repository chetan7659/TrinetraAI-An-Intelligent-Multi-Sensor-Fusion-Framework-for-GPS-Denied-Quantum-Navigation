"""Canonical domain representation of a single synchronized sensor frame.

This module defines :class:`SensorRecord`, the **only** type that the Domain
layer and Application layer ever receive from the data pipeline.

It is completely independent of:

* HDF5 files or any binary format
* RoNIN, TLIO, EuRoC, KITTI, or any other specific dataset
* ``info.json``, YAML, CSV, or any text format

Every dataset's :class:`CanonicalMapper` produces ``SensorRecord`` objects.
The Domain layer processes ``SensorRecord`` objects. Neither layer knows about
the other's internals.

Design contract
---------------
* **Frozen** (immutable) — domain state must never be modified after creation.
* **No methods** — pure value object; all logic lives in the domain services.
* **No dataset-specific imports** — this module has zero adapter dependencies.
* **Typed fields** — all sensor streams are explicitly typed; no ``Any``.

Canonical field definitions
---------------------------
All quantities are expressed in SI units, in the **body frame** of the IMU
device (as stored in the dataset's synchronized stream), unless otherwise
documented by the dataset's Canonical Mapper.

+------------------------+----------------+-----------------------------------+
| Field                  | Unit           | Description                       |
+========================+================+===================================+
| ``frame_id``           | —              | Zero-based index within the       |
|                        |                | recording (after start_frame      |
|                        |                | offset has been applied)          |
+------------------------+----------------+-----------------------------------+
| ``timestamp``          | seconds        | IMU system time                   |
+------------------------+----------------+-----------------------------------+
| ``accelerometer``      | m/s²           | Total specific force (x, y, z)    |
+------------------------+----------------+-----------------------------------+
| ``gyroscope``          | rad/s          | Angular velocity, calibrated      |
|                        |                | (x, y, z)                        |
+------------------------+----------------+-----------------------------------+
| ``linear_acceleration``| m/s²           | Specific force with gravity       |
|                        |                | component removed (x, y, z)       |
+------------------------+----------------+-----------------------------------+
| ``gravity``            | m/s²           | Estimated gravity vector          |
|                        |                | (x, y, z)                        |
+------------------------+----------------+-----------------------------------+
| ``orientation``        | —              | Orientation estimate as unit      |
|                        |                | quaternion (x, y, z, w)          |
+------------------------+----------------+-----------------------------------+
| ``magnetometer``       | µT             | Magnetic field (x, y, z)         |
+------------------------+----------------+-----------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Type aliases (mirrors raw_frames.py conventions)
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


@dataclass(frozen=True)
class SensorRecord:
    """Canonical, immutable representation of one synchronized IMU frame.

    This is the **domain boundary type** produced by every
    :class:`CanonicalMapper` implementation and consumed by the Application
    layer and Domain layer.

    Attributes:
        frame_id: Zero-based index within the recording after the
            ``start_frame`` offset has been applied.
        timestamp: IMU system time in seconds.
        accelerometer: Total specific force in m/s², axes (x, y, z).
            Includes gravity component.
        gyroscope: Calibrated angular velocity in rad/s, axes (x, y, z).
        linear_acceleration: Specific force with gravity removed, in m/s²,
            axes (x, y, z).
        gravity: Estimated gravity vector in m/s², axes (x, y, z).
        orientation: Orientation estimate as a unit quaternion (x, y, z, w).
            Derived from the game rotation vector (accelerometer + gyroscope
            fusion; magnetometer-free) for maximum portability across datasets.
        magnetometer: Magnetic field reading in µT, axes (x, y, z).

    Notes:
        This dataclass is *frozen* (immutable).  No methods and no
        dataset-specific logic live here — it is a canonical value object.
        See :class:`~trinetra.adapters.datasets.ronin.canonical_mapper.RoninCanonicalMapper`
        for the RoNIN-specific implementation.
    """

    # -- Identity -------------------------------------------------------------
    frame_id: int

    # -- Time -----------------------------------------------------------------
    timestamp: float  # seconds

    # -- Inertial sensors -----------------------------------------------------
    accelerometer: Vec3  # m/s² (x, y, z) — total specific force
    gyroscope: Vec3  # rad/s (x, y, z) — calibrated

    # -- Derived inertial -------------------------------------------------
    linear_acceleration: Vec3  # m/s² (x, y, z) — gravity removed
    gravity: Vec3  # m/s² (x, y, z) — gravity vector

    # -- Orientation ----------------------------------------------------------
    orientation: Vec4  # quaternion (x, y, z, w) — game rotation vector

    # -- Magnetic field -------------------------------------------------------
    magnetometer: Vec3  # µT (x, y, z)
