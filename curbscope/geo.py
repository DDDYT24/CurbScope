"""Rigid transforms between sensor frames.

All transforms are 4x4 homogeneous matrices named T_<to>_<from>, so
T_w_cam2 @ p_cam2 gives the point in world coordinates. This follows the
KITTI devkit and pykitti naming.
"""

import numpy as np


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, using R^T instead of a general matrix inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply T to an (N, 3) array of points and return (N, 3)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


def world_from_cam2(T_w_imu: np.ndarray, T_cam2_imu: np.ndarray) -> np.ndarray:
    """T_w_cam2 = T_w_imu @ inv(T_cam2_imu)."""
    return T_w_imu @ invert_transform(T_cam2_imu)


def world_from_velo(T_w_imu: np.ndarray, T_velo_imu: np.ndarray) -> np.ndarray:
    """T_w_velo = T_w_imu @ inv(T_velo_imu)."""
    return T_w_imu @ invert_transform(T_velo_imu)
