import numpy as np

from curbscope import geo


def random_transform(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = q
    T[:3, 3] = rng.normal(scale=10, size=3)
    return T


def test_invert_matches_general_inverse():
    T = random_transform(np.random.default_rng(0))
    np.testing.assert_allclose(geo.invert_transform(T), np.linalg.inv(T), atol=1e-12)


def test_round_trip_returns_original_points():
    rng = np.random.default_rng(1)
    T = random_transform(rng)
    points = rng.normal(scale=20, size=(50, 3))
    back = geo.apply_transform(geo.invert_transform(T), geo.apply_transform(T, points))
    np.testing.assert_allclose(back, points, atol=1e-9)


def test_velo_to_world_agrees_through_cam2():
    """velo -> cam2 -> world must equal velo -> world for the same calibration."""
    rng = np.random.default_rng(2)
    T_w_imu = random_transform(rng)
    T_cam2_velo = random_transform(rng)
    T_velo_imu = random_transform(rng)
    T_cam2_imu = T_cam2_velo @ T_velo_imu
    points_velo = rng.normal(scale=15, size=(20, 3))

    via_cam2 = geo.apply_transform(
        geo.world_from_cam2(T_w_imu, T_cam2_imu), geo.apply_transform(T_cam2_velo, points_velo)
    )
    direct = geo.apply_transform(geo.world_from_velo(T_w_imu, T_velo_imu), points_velo)
    np.testing.assert_allclose(via_cam2, direct, atol=1e-9)


def test_identity_pose_keeps_points():
    points = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(geo.apply_transform(np.eye(4), points), points)
