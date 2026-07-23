"""Cross-sensor frame consistency and residual validation."""

from __future__ import annotations

import numpy as np


def validate_sensor_frame_consistency(
    accel: np.ndarray | list[tuple[float, ...]],
    gravity: np.ndarray | list[tuple[float, ...]],
    linear_accel: np.ndarray | list[tuple[float, ...]],
) -> dict[str, float]:
    """Verify physical relation Accelerometer = Gravity + Linear Acceleration.

    Computes residual R = Accelerometer - (Gravity + Linear Acceleration)
    and reports stats on the norm: mean, std, min, max, 95th percentile, and RMS.

    Args:
        accel: Array or list of accelerometer vectors.
        gravity: Array or list of gravity vectors.
        linear_accel: Array or list of linear acceleration vectors.

    Returns:
        Dictionary of consistency metrics.
    """
    if len(accel) == 0 or len(gravity) == 0 or len(linear_accel) == 0:
        return {
            "mean_residual": 0.0,
            "std_residual": 0.0,
            "min_residual": 0.0,
            "max_residual": 0.0,
            "p95_residual": 0.0,
            "rms_residual": 0.0,
        }

    a = np.array(accel)
    g = np.array(gravity)
    l_acc = np.array(linear_accel)

    # Filter NaNs/Infs
    mask = np.isfinite(a).all(axis=1) & np.isfinite(g).all(axis=1) & np.isfinite(l_acc).all(axis=1)

    valid_a = a[mask]
    valid_g = g[mask]
    valid_l = l_acc[mask]

    if len(valid_a) == 0:
        return {
            "mean_residual": 0.0,
            "std_residual": 0.0,
            "min_residual": 0.0,
            "max_residual": 0.0,
            "p95_residual": 0.0,
            "rms_residual": 0.0,
        }

    # Compute residual: R = A - (G + L)
    residuals = valid_a - (valid_g + valid_l)
    residual_norms = np.linalg.norm(residuals, axis=1)

    mean_res = float(np.mean(residual_norms))
    std_res = float(np.std(residual_norms))
    min_res = float(np.min(residual_norms))
    max_res = float(np.max(residual_norms))
    p95_res = float(np.percentile(residual_norms, 95))
    rms_res = float(np.sqrt(np.mean(residual_norms**2)))

    return {
        "mean_residual": mean_res,
        "std_residual": std_res,
        "min_residual": min_res,
        "max_residual": max_res,
        "p95_residual": p95_res,
        "rms_residual": rms_res,
    }
