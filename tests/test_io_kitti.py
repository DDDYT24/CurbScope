from pathlib import Path

import numpy as np
import pytest

from curbscope import io_kitti

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_drive_id():
    assert io_kitti.parse_drive_id("2011_09_26_drive_0019") == ("2011_09_26", "0019")


@pytest.mark.parametrize("bad", ["0019", "2011_09_26_drive_19", "2011_09_26_drive_0019_sync"])
def test_parse_drive_id_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        io_kitti.parse_drive_id(bad)


def test_drive_dir_layout(tmp_path):
    expected = tmp_path / "2011_09_26" / "2011_09_26_drive_0019_sync"
    assert io_kitti.drive_dir(tmp_path, "2011_09_26_drive_0019") == expected


def test_open_drive_missing_folder_points_to_docs(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_kitti"):
        io_kitti.open_drive(tmp_path, "2011_09_26_drive_0019")


def test_load_tracklets():
    tracklets = io_kitti.load_tracklets(FIXTURES / "tracklets_small.xml")
    assert [t.object_type for t in tracklets] == ["Car", "Pedestrian"]
    car = tracklets[0]
    assert (car.h, car.w, car.l) == (1.5, 1.6, 4.2)
    assert car.poses.shape == (2, 6)
    np.testing.assert_array_equal(car.frames, [3, 4])
    np.testing.assert_allclose(car.poses[1, :3], [9.0, -3.0, -1.7])
    assert tracklets[1].poses.shape == (1, 6)


def test_open_fake_drive_with_pykitti(fake_kitti):
    raw = io_kitti.open_drive(fake_kitti, "2011_09_26_drive_0001")
    T_w_imu = io_kitti.poses(raw)
    assert T_w_imu.shape == (3, 4, 4)
    np.testing.assert_allclose(T_w_imu[0], np.eye(4), atol=1e-6)  # world starts at frame 0
    np.testing.assert_allclose(io_kitti.timestamps_s(raw), [0.0, 0.1, 0.2], atol=1e-6)

    calib = io_kitti.calibration(raw)
    assert set(calib) == {"T_cam2_velo", "P_rect_20", "T_cam2_imu", "T_velo_imu"}
    np.testing.assert_allclose(calib["T_cam2_imu"], calib["T_cam2_velo"] @ calib["T_velo_imu"])

    # 1e-5 degree of latitude is about 1.11 m, two steps
    summary = io_kitti.drive_summary(io_kitti.timestamps_s(raw), T_w_imu[:, :3, 3])
    assert summary["distance_m"] == pytest.approx(2 * 1.11, rel=0.02)


def test_drive_summary_straight_line():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    positions = np.array([[0, 0, 0], [3, 4, 0], [6, 8, 1], [9, 12, 0]], dtype=float)
    summary = io_kitti.drive_summary(t, positions)
    assert summary["frames"] == 4
    assert summary["duration_s"] == pytest.approx(0.3)
    assert summary["distance_m"] == pytest.approx(15.0)  # height changes are ignored
