"""Geometry: rigid transforms, LiDAR-to-image projection, 3D position of a box, lat/lon.

All transforms are 4x4 homogeneous matrices named T_<to>_<from>, so
T_w_cam2 @ p_cam2 gives the point in world coordinates. This follows the
KITTI devkit and pykitti naming.

Frames: velo (x forward, y left, z up), cam2 (x right, y down, z forward),
world (pykitti's local metric frame: x east, y north, z up, origin at the first OXTS fix).
"""

import numpy as np
from pyproj import Proj

EARTH_RADIUS_M = 6378137.0  # the sphere radius pykitti uses for its Mercator poses


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


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel coordinates (N, 2) of cam2 points. Only meaningful for points with z > 0.

    K is P_rect_20[:3, :3]: pykitti's T_cam2_velo already includes the rectifying rotation
    and cam2's offset, so the intrinsics alone finish the projection.
    """
    uvw = points_cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def frame_points(
    velo: np.ndarray, T_cam2_velo: np.ndarray, K: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """One scan's above-ground points in front of the camera: cam2 xyz (N, 3) and pixels (N, 2).

    cfg keys: lidar_height_m, min_height_above_ground_m. The ground test uses the Velodyne z,
    so it assumes a flat road. Done once per frame, then box_points() just selects.
    """
    xyz = velo[:, :3]
    above = xyz[:, 2] > -cfg["lidar_height_m"] + cfg["min_height_above_ground_m"]
    cam = apply_transform(T_cam2_velo, xyz[above])
    cam = cam[cam[:, 2] > 0.5]  # in front of the camera
    return cam, project(cam, K)


def box_points(cam: np.ndarray, uv: np.ndarray, box: tuple, keep_fraction: float) -> np.ndarray:
    """cam2 points (M, 3) whose pixels fall in the central part of an image box.

    box is (x1, y1, x2, y2) in pixels; keep_fraction of its width and height is kept,
    centred, so points on the background around the object are less likely.
    """
    x1, y1, x2, y2 = box
    margin = (1 - keep_fraction) / 2
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    inside = (
        (uv[:, 0] >= x1 + dx) & (uv[:, 0] <= x2 - dx)
        & (uv[:, 1] >= y1 + dy) & (uv[:, 1] <= y2 - dy)
    )
    return cam[inside]


def nearest_cluster(points: np.ndarray, gap_m: float, min_points: int) -> np.ndarray | None:
    """Nearest group of points along the viewing ray.

    Sort by distance from the camera and cut wherever two neighbours are more than gap_m
    apart. Return the first group with at least min_points (smaller groups in front are
    treated as noise, e.g. a branch), or None.
    """
    if len(points) == 0:
        return None
    dist = np.linalg.norm(points, axis=1)
    order = np.argsort(dist)
    cuts = np.flatnonzero(np.diff(dist[order]) > gap_m) + 1
    for group in np.split(order, cuts):
        if len(group) >= min_points:
            return points[group]
    return None


def locate(points_cam: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, int]:
    """3D position (cam2 frame) of the object seen in a box, and the number of box points.

    Unknown (None) when the box has fewer than min_box_points or no cluster is big enough.
    """
    n = len(points_cam)
    if n < cfg["min_box_points"]:
        return None, n
    cluster = nearest_cluster(points_cam, cfg["cluster_gap_m"], cfg["min_cluster_points"])
    if cluster is None:
        return None, n
    return np.median(cluster, axis=0), n


def _mercator(origin_lat: float) -> Proj:
    """Spherical Mercator scaled by cos(origin latitude), exactly as pykitti builds its poses."""
    return Proj(proj="merc", a=EARTH_RADIUS_M, b=EARTH_RADIUS_M,
                k_0=np.cos(np.radians(origin_lat)), lon_0=0)


def world_to_latlon(xy: np.ndarray, origin_lat: float, origin_lon: float) -> np.ndarray:
    """World x, y (N, 2) in metres -> (N, 2) latitude, longitude in degrees."""
    proj = _mercator(origin_lat)
    x0, y0 = proj(origin_lon, origin_lat)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    lon, lat = proj(xy[:, 0] + x0, xy[:, 1] + y0, inverse=True)
    return np.column_stack([lat, lon])


def latlon_to_world(latlon: np.ndarray, origin_lat: float, origin_lon: float) -> np.ndarray:
    """Inverse of world_to_latlon."""
    proj = _mercator(origin_lat)
    x0, y0 = proj(origin_lon, origin_lat)
    latlon = np.asarray(latlon, dtype=float).reshape(-1, 2)
    x, y = proj(latlon[:, 1], latlon[:, 0])
    return np.column_stack([x - x0, y - y0])
