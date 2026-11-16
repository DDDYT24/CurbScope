import numpy as np
import pandas as pd
import pytest

from curbscope import geo_eval
from curbscope.io_kitti import Tracklet


def test_match_is_global_and_gated():
    det = np.array([[0.0, 0.0], [3.0, 0.0], [50.0, 0.0]])
    gt = np.array([[1.0, 0.0], [4.0, 0.0]])
    pairs = geo_eval.match(det, gt, max_m=4.0)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (1, 1)]  # det 2 is beyond the gate
    assert geo_eval.match(det, np.zeros((0, 2)), 4.0) == []


def test_footprint_distance():
    # 4 m long (along yaw), 2 m wide box at the origin
    assert geo_eval.footprint_distance((1.5, 0.5), (0, 0), 0.0, 4.0, 2.0) == 0.0
    assert geo_eval.footprint_distance((3.0, 0.0), (0, 0), 0.0, 4.0, 2.0) == pytest.approx(1.0)
    # turned 90 degrees: the long side now runs along y
    turned = np.pi / 2
    assert geo_eval.footprint_distance((0.0, 3.0), (0, 0), turned, 4.0, 2.0) == pytest.approx(1.0)
    assert geo_eval.footprint_distance((3.0, 0.0), (0, 0), turned, 4.0, 2.0) == pytest.approx(2.0)


def test_tracklets_by_frame_filters_and_moves_to_world():
    poses = np.array([[10.0, 2.0, -1.7, 0, 0, 0.1], [11.0, 2.0, -1.7, 0, 0, 0.1]])
    car = Tracklet("Car", 1.5, 1.6, 4.0, first_frame=1, poses=poses)
    person = Tracklet("Pedestrian", 1.8, 0.6, 0.8, first_frame=0, poses=poses[:1])
    T = np.tile(np.eye(4), (2, 1, 1))  # only frames 0 and 1 exist
    T[1, :2, 3] = [100.0, 50.0]
    labels = geo_eval.tracklets_by_frame([car, person], T, ["Car"])
    assert labels["frame"].tolist() == [1]  # frame 2 is past the drive, person filtered out
    row = labels.iloc[0]
    assert (row["x"], row["y"]) == (110.0, 52.0)
    assert row["range_m"] == pytest.approx(np.hypot(10, 2))
    assert row["yaw"] == pytest.approx(0.1)


def test_match_drive_and_error_by_range():
    located = pd.DataFrame({"frame": [0, 0, 1], "track_id": [1, 2, 1],
                            "x": [10.0, 30.0, 11.0], "y": [0.0, 0.0, 0.0]})
    labels = pd.DataFrame({"frame": [0, 0, 1], "tracklet": [0, 1, 0],
                           "x": [12.0, 31.0, 12.0], "y": [0.0, 0.0, 0.0],
                           "yaw": [0.0, 0.0, 0.0], "l": [4.0, 4.0, 4.0], "w": [2.0, 2.0, 2.0],
                           "range_m": [5.0, 25.0, 6.0]})
    pairs = geo_eval.match_drive(located, labels, max_m=4.0)
    assert len(pairs) == 3
    assert pairs["center_error_m"].tolist() == pytest.approx([2.0, 1.0, 1.0])
    assert pairs["footprint_error_m"].tolist() == pytest.approx([0.0, 0.0, 0.0])

    stats = geo_eval.error_by_range(pairs, [0, 10, 20, 40])
    assert stats["0-10 m"]["pairs"] == 2 and stats["0-10 m"]["center"]["median"] == 1.5
    assert stats["10-20 m"] == {"pairs": 0, "center": {"median": None, "p90": None},
                                "footprint": {"median": None, "p90": None}}
    assert stats["20-40 m"]["pairs"] == 1 and stats["40+ m"]["pairs"] == 0
    assert stats["all"]["pairs"] == 3


def test_box_corners_velo():
    corners = geo_eval.box_corners_velo(np.array([10.0, 0, -1.7, 0, 0, 0]), h=1.5, w=2.0,
                                        length=4.0)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners[:, 0].min(), 8.0)
    np.testing.assert_allclose(corners[:, 0].max(), 12.0)
    np.testing.assert_allclose(sorted(set(np.round(corners[:, 2], 6))), [-1.7, -0.2])
