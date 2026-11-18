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


# --- M2: projection, box points, clustering, lat/lon -------------------------------------

K = np.array([[700.0, 0, 600], [0, 700.0, 180], [0, 0, 1]])
# Velodyne (x forward, y left, z up) -> camera (x right, y down, z forward), no offset.
T_CAM_VELO = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
GEO_CFG = {"box_keep_fraction": 0.6, "lidar_height_m": 1.73, "min_height_above_ground_m": 0.3,
           "cluster_gap_m": 0.5, "min_cluster_points": 5, "min_box_points": 10}


def test_project_principal_point_and_offset():
    uv = geo.project(np.array([[0.0, 0.0, 10.0], [1.0, -0.5, 10.0]]), K)
    np.testing.assert_allclose(uv, [[600, 180], [670, 145]])


def test_frame_points_drops_ground_and_points_behind():
    velo = np.array([
        [10.0, 0.0, 0.0, 0],    # kept: 1.73 m above the road
        [10.0, 0.0, -1.6, 0],   # dropped: 0.13 m above the road
        [-10.0, 0.0, 0.0, 0],   # dropped: behind the camera
    ])
    cam, uv = geo.frame_points(velo, T_CAM_VELO, K, GEO_CFG)
    np.testing.assert_allclose(cam, [[0.0, 0.0, 10.0]])
    np.testing.assert_allclose(uv, [[600, 180]])


def test_box_points_keeps_only_central_part():
    uv = np.array([[105.0, 150.0], [150.0, 150.0], [195.0, 150.0]])
    cam = np.arange(9, dtype=float).reshape(3, 3)
    # box 100..200 wide, central 60% is 120..180: only the middle point survives
    kept = geo.box_points(cam, uv, (100, 100, 200, 200), keep_fraction=0.6)
    np.testing.assert_allclose(kept, cam[[1]])


def test_nearest_cluster_skips_small_noise_in_front():
    rng = np.random.default_rng(0)
    noise = np.array([[0, 0, 5.0], [0, 0, 5.1]])                       # 2 points: a branch
    car = np.column_stack([rng.normal(0, 0.3, 8), rng.normal(0, 0.3, 8), 10 + rng.random(8) * 0.3])
    wall = np.column_stack([np.zeros(20), np.zeros(20), np.linspace(20, 20.4, 20)])
    cluster = geo.nearest_cluster(np.vstack([wall, noise, car]), gap_m=0.5, min_points=5)
    assert len(cluster) == 8 and cluster[:, 2].max() < 11
    assert geo.nearest_cluster(noise, gap_m=0.5, min_points=5) is None
    assert geo.nearest_cluster(np.zeros((0, 3)), gap_m=0.5, min_points=5) is None


def test_locate_unknown_with_too_few_points_else_median():
    few = np.tile([0.0, 0.0, 10.0], (9, 1))
    assert geo.locate(few, GEO_CFG) == (None, 9)
    pts = np.column_stack([np.zeros(11), np.zeros(11), np.linspace(10, 10.4, 11)])
    pos, n = geo.locate(pts, GEO_CFG)
    assert n == 11
    np.testing.assert_allclose(pos, [0, 0, 10.2])


def test_latlon_round_trip():
    xy = np.array([[0.0, 0.0], [500.0, -300.0], [-1200.0, 800.0]])
    latlon = geo.world_to_latlon(xy, 49.03, 8.44)
    np.testing.assert_allclose(latlon[0], [49.03, 8.44], atol=1e-12)
    np.testing.assert_allclose(geo.latlon_to_world(latlon, 49.03, 8.44), xy, atol=1e-6)


def test_latlon_matches_pykitti_poses(fake_kitti):
    """Converting each OXTS fix to world must land on pykitti's own pose translation."""
    from curbscope import io_kitti

    raw = io_kitti.open_drive(fake_kitti, "2011_09_26_drive_0001")
    fixes = np.array([[o.packet.lat, o.packet.lon] for o in raw.oxts])
    xy = geo.latlon_to_world(fixes, fixes[0, 0], fixes[0, 1])
    np.testing.assert_allclose(xy, io_kitti.poses(raw)[:, :2, 3], atol=1e-6)


def test_latlon_single_point_matches_batch():
    one = geo.world_to_latlon(np.array([[10.0, 3.0]]), 49.0, 8.4)
    many = geo.world_to_latlon(np.array([[10.0, 3.0], [0.0, 0.0]]), 49.0, 8.4)
    np.testing.assert_allclose(one[0], many[0])
    np.testing.assert_allclose(geo.latlon_to_world(one, 49.0, 8.4), [[10.0, 3.0]], atol=1e-6)
