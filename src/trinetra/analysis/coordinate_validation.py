"""Coordinate consistency and physical vector validation."""

from __future__ import annotations

import numpy as np


def validate_gravity(gravity_vectors: np.ndarray | list[tuple[float, ...]]) -> dict[str, float]:
    """Validate gravity vector magnitude, deviation, and bias relative to 9.81 m/s^2.

    Args:
        gravity_vectors: Array or list of gravity vectors.

    Returns:
        Dictionary of magnitude statistics and bias.
    """
    if len(gravity_vectors) == 0:
        return {
            "mean_magnitude": 0.0,
            "std_magnitude": 0.0,
            "min_magnitude": 0.0,
            "max_magnitude": 0.0,
            "bias": 0.0,
        }

    arr = np.array(gravity_vectors)
    # Filter out NaNs/Infs
    mask = np.isfinite(arr).all(axis=1)
    valid_arr = arr[mask]

    if len(valid_arr) == 0:
        return {
            "mean_magnitude": 0.0,
            "std_magnitude": 0.0,
            "min_magnitude": 0.0,
            "max_magnitude": 0.0,
            "bias": 0.0,
        }

    mags = np.linalg.norm(valid_arr, axis=1)
    mean_mag = float(np.mean(mags))
    bias = mean_mag - 9.81

    return {
        "mean_magnitude": mean_mag,
        "std_magnitude": float(np.std(mags)),
        "min_magnitude": float(np.min(mags)),
        "max_magnitude": float(np.max(mags)),
        "bias": bias,
    }


def check_coordinate_consistency(
    accel_vectors: np.ndarray | list[tuple[float, ...]],
    gravity_vectors: np.ndarray | list[tuple[float, ...]],
) -> dict[str, float]:
    """Validate axis consistency and alignment of physical vectors.

    Args:
        accel_vectors: Array or list of accelerometer vectors.
        gravity_vectors: Array or list of gravity vectors.

    Returns:
        Dictionary indicating alignment statistics and check results.
    """
    if len(accel_vectors) == 0 or len(gravity_vectors) == 0:
        return {
            "accel_mean_magnitude": 0.0,
            "gravity_alignment_axis": -1,
            "axis_polarities_consistent": 1.0,
        }

    acc_arr = np.array(accel_vectors)
    grav_arr = np.array(gravity_vectors)

    # Filter NaNs/Infs
    mask = np.isfinite(acc_arr).all(axis=1) & np.isfinite(grav_arr).all(axis=1)
    valid_acc = acc_arr[mask]
    valid_grav = grav_arr[mask]

    if len(valid_acc) == 0:
        return {
            "accel_mean_magnitude": 0.0,
            "gravity_alignment_axis": -1,
            "axis_polarities_consistent": 1.0,
        }

    acc_mags = np.linalg.norm(valid_acc, axis=1)
    mean_acc = float(np.mean(acc_mags))

    # Identify primary gravity alignment axis (0=X, 1=Y, 2=Z) using absolute mean magnitude
    mean_grav_abs = np.mean(np.abs(valid_grav), axis=0)
    primary_axis = int(np.argmax(mean_grav_abs))

    # Check if the sign of gravity along the primary axis remains consistent
    polarities = np.sign(valid_grav[:, primary_axis])
    unique_polarities = np.unique(polarities[polarities != 0.0])
    # If the polarity flips flags (outside sensor noise / dynamic motion), consistency is low
    polarities_consistent = 1.0 if len(unique_polarities) <= 1 else 0.0

    return {
        "accel_mean_magnitude": mean_acc,
        "gravity_alignment_axis": primary_axis,
        "axis_polarities_consistent": polarities_consistent,
    }
