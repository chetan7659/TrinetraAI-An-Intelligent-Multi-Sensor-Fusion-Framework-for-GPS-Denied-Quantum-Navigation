"""Immutable data models representing the contents of a RoNIN ``info.json`` file.

This module contains only value objects (frozen dataclasses). It has no file I/O,
no business logic, and no external dependencies beyond the Python standard library.

Schema source: README.txt (JSON data description section) and direct inspection of
real ``info.json`` files from the RoNIN dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImuCalibration:
    """IMU sensor calibration parameters from ``info.json``.

    These values represent the calibration state of the IMU device at the
    time of data collection.  All calibration parameters are required fields
    that exist in every ``info.json`` file observed in the dataset.

    Attributes:
        imu_init_gyro_bias: Gyroscope bias vector (x, y, z) in rad/s,
            estimated at the beginning of the recording session.
        imu_end_gyro_bias: Gyroscope bias vector (x, y, z) in rad/s,
            estimated at the end of the recording session.
        imu_acce_bias: Accelerometer bias vector (x, y, z) in m/s².
        imu_acce_scale: Accelerometer scale factor vector (x, y, z).
            Values near 1.0 indicate well-calibrated axes.
    """

    imu_init_gyro_bias: tuple[float, float, float]
    imu_end_gyro_bias: tuple[float, float, float]
    imu_acce_bias: tuple[float, float, float]
    imu_acce_scale: tuple[float, float, float]


@dataclass(frozen=True)
class TimeSynchronization:
    """Time synchronisation parameters between the Tango and IMU devices.

    The formula for converting system timestamps between the two devices is::

        t_imu = t_tango - tango_reference_time + imu_reference_time + imu_time_offset

    Attributes:
        imu_reference_time: Approximate IMU time alignment anchor (nanoseconds).
        tango_reference_time: Approximate Tango time alignment anchor (nanoseconds).
        imu_time_offset: Fine-tuned time alignment correction (seconds).
    """

    imu_reference_time: float
    tango_reference_time: float
    imu_time_offset: float


@dataclass(frozen=True)
class OrientationCalibration:
    """Calibration quaternions aligning the Tango and IMU coordinate frames.

    Attributes:
        align_tango_to_body: Quaternion (x, y, z, w) representing the
            approximate alignment between the Tango device coordinate frame
            and the body frame of the subject.
        start_calibration: Quaternion (w, x, y, z) alignment between the
            Tango reference frame and the IMU game rotation vector at the
            start of the recording.
        end_calibration: Quaternion (w, x, y, z) alignment computed at
            the end of the recording.
    """

    align_tango_to_body: tuple[float, float, float, float]
    start_calibration: tuple[float, float, float, float]
    end_calibration: tuple[float, float, float, float]


@dataclass(frozen=True)
class OrientationErrors:
    """Orientation drift / error metrics for the recording.

    These values quantify how well the sensor fusion algorithms performed
    over the course of the recording.

    Attributes:
        gyro_integration_error: Cumulative orientation drift from pure gyro
            integration (degrees).
        grv_ori_error: Orientation error from the Android Game Rotation Vector
            sensor relative to the Tango ground truth (degrees).
        ekf_ori_error: Orientation error from the Extended Kalman Filter
            relative to the Tango ground truth (degrees).
    """

    gyro_integration_error: float
    grv_ori_error: float
    ekf_ori_error: float


@dataclass(frozen=True)
class RoninRecordingMetadata:
    """Complete, immutable metadata for one RoNIN recording session.

    This is the top-level value object produced by :class:`RoninMetadataLoader`
    when it parses an ``info.json`` file.  Every field maps directly to a key
    in the real ``info.json`` schema; no fields are invented.

    Attributes:
        recording_id: The name of the recording directory (e.g. ``"a000_1"``).
            Injected by the loader from the :class:`~.models.Recording` object;
            not present inside ``info.json`` itself.
        split: The dataset split the recording belongs to (e.g.
            ``"train_dataset_1"``).  Injected from the
            :class:`~.models.Recording` object.
        length: Recording duration in seconds.
        date: Collection date as a string in ``mm/dd/yy`` format.
        device: Device identifier string (e.g. ``"asus5"``, ``"samsung1"``).
        recording_type: Annotation type (e.g. ``"annotated"``).  Stored as
            ``recording_type`` to avoid shadowing the Python built-in ``type``.
        start_frame: Index of the first usable data frame, marking the end of
            the pre-calibration and synchronisation period.
        calibration: IMU sensor calibration parameters.
        time_sync: Time synchronisation parameters between the two devices.
        orientation: Orientation alignment quaternions.
        errors: Orientation drift / error metrics.
    """

    recording_id: str
    split: str

    # ── Sequence-level metadata ───────────────────────────────────────────────
    length: float
    date: str
    device: str
    recording_type: str
    start_frame: int

    # ── Calibration and synchronisation ──────────────────────────────────────
    calibration: ImuCalibration
    time_sync: TimeSynchronization
    orientation: OrientationCalibration
    errors: OrientationErrors
