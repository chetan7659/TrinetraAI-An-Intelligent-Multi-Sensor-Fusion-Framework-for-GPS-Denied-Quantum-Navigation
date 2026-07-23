"""Quaternion validation, sign-flips, and angular continuity validation."""

from __future__ import annotations

import numpy as np

# Configurable default threshold for angular steps
DEFAULT_MAX_ANGULAR_STEP_RAD = np.deg2rad(15.0)


def validate_quaternions(
    quats: np.ndarray | list[tuple[float, ...]],
    max_angular_step_rad: float = DEFAULT_MAX_ANGULAR_STEP_RAD,
) -> dict[str, float]:
    """Verify quaternion properties: norms, sign-flips, and angular velocity smoothness.

    Args:
        quats: Array or list of quaternions (x, y, z, w).
        max_angular_step_rad: Configurable threshold for detecting angular jumps.

    Returns:
        Dictionary of orientation consistency metrics.
    """
    if len(quats) == 0:
        return {
            "min_norm": 0.0,
            "max_norm": 0.0,
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "nan_inf_count": 0.0,
            "sign_flip_count": 0.0,
            "abnormal_jump_count": 0.0,
            "mean_angular_step_rad": 0.0,
            "max_angular_step_rad": 0.0,
        }

    arr = np.array(quats)

    # NaN / Inf detection
    nan_inf_mask = ~np.isfinite(arr).all(axis=1)
    nan_inf_count = int(np.sum(nan_inf_mask))

    # Filter valid quaternions
    valid_arr = arr[~nan_inf_mask]

    if len(valid_arr) == 0:
        return {
            "min_norm": 0.0,
            "max_norm": 0.0,
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "nan_inf_count": float(nan_inf_count),
            "sign_flip_count": 0.0,
            "abnormal_jump_count": 0.0,
            "mean_angular_step_rad": 0.0,
            "max_angular_step_rad": 0.0,
        }

    # Normalize stats
    norms = np.linalg.norm(valid_arr, axis=1)

    # Consecutive check
    sign_flips = 0
    abnormal_jumps = 0
    angular_steps = []

    if len(valid_arr) > 1:
        # Compute consecutive dot products
        # dot_products[i] = q[i] . q[i+1]
        dot_products = np.sum(valid_arr[:-1] * valid_arr[1:], axis=1)

        # Normalize the vectors to avoid floating point issues during arccos
        norm_product = norms[:-1] * norms[1:]
        # Avoid division by zero
        norm_product[norm_product == 0.0] = 1e-9
        cos_theta = dot_products / norm_product

        # Clip to ensure valid values for arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Sign flips
        sign_flips = int(np.sum(cos_theta < 0))

        # Compute relative rotation angle (rectifying sign-flips)
        abs_cos_theta = np.abs(cos_theta)
        relative_angles = 2.0 * np.arccos(abs_cos_theta)

        angular_steps = list(relative_angles)
        abnormal_jumps = int(np.sum(relative_angles > max_angular_step_rad))

    mean_step = float(np.mean(angular_steps)) if angular_steps else 0.0
    max_step = float(np.max(angular_steps)) if angular_steps else 0.0

    return {
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "nan_inf_count": float(nan_inf_count),
        "sign_flip_count": float(sign_flips),
        "abnormal_jump_count": float(abnormal_jumps),
        "mean_angular_step_rad": mean_step,
        "max_angular_step_rad": max_step,
    }
