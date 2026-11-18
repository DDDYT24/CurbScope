"""Curb segments on synthetic geometry."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curbscope import segments as sg

ROOT = Path(__file__).parents[1]
LEN_CFG = {"vehicle_length_range_m": [2.0, 8.0], "min_length_points": 20,
           "default_vehicle_length_m": 4.5}


def test_interval_union_clip_subtract_length():
    assert sg.union([(5, 7), (0, 2), (1, 3), (3, 4), (8, 8)]) == [(0, 4), (5, 7)]
    assert sg.clip([(0, 10), (15, 25)], 5, 20) == [(5, 10), (15, 20)]
    assert sg.subtract([(0, 20)], [(5, 8), (7, 10), (18, 30)]) == [(0, 5), (10, 18)]
    assert sg.subtract([(0, 5)], [(5, 9)]) == [(0, 5)]
    assert sg.length([(0, 4), (2, 6), (10, 11)]) == 7.0


def test_place_side_and_distance_on_straight_path():
    line = sg.path_line(np.array([[0.0, 0.0], [100.0, 0.0]]))
    s, offset, tangent = sg.place(line, (30.0, 5.0))
    assert s == pytest.approx(30) and offset == pytest.approx(5)  # left of travel
    np.testing.assert_allclose(tangent, [1, 0])
    assert sg.place(line, (30.0, -4.0))[1] == pytest.approx(-4)  # right


def test_place_after_a_left_turn():
    # drive east 50 m, then north 50 m; a point west of the northbound leg is on the left
    line = sg.path_line(np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]))
    s, offset, tangent = sg.place(line, (47.0, 30.0))
    assert s == pytest.approx(80) and offset == pytest.approx(3)
    np.testing.assert_allclose(tangent, [0, 1], atol=1e-9)


def test_vehicle_length_from_lidar_or_default():
    rng = np.random.default_rng(0)
    along = rng.uniform(0, 4.0, 200)
    points = np.column_stack([along, rng.normal(0, 0.2, 200)])
    length, source = sg.vehicle_length(points, np.array([1.0, 0.0]), LEN_CFG)
    assert source == "lidar" and length == pytest.approx(3.6, abs=0.25)  # 5th to 95th pct
    assert sg.vehicle_length(points[:10], np.array([1.0, 0.0]), LEN_CFG) == (4.5, "default")
    long = np.column_stack([np.linspace(0, 12, 50), np.zeros(50)])
    assert sg.vehicle_length(long, np.array([1.0, 0.0]), LEN_CFG) == (4.5, "default")


def segments_fixture():
    cars = pd.DataFrame({
        "track_id": [7, 3, 9], "side": ["left", "left", "right"],
        "s": [10.0, 14.5, 30.0], "length_m": [4.0, 4.0, 5.0],
        "first_seen_s": [12.0, 2.5, 4.0]})
    hydrant = pd.DataFrame({"side": ["left"], "s": [18.0], "half_width_m": [5.0]})
    return sg.build_segments(40.0, 20.0, cars, hydrant)


def test_build_segments_lengths_buffers_and_occupancy():
    segs = segments_fixture().set_index(["side", "start_m"])
    first = segs.loc[("left", 0.0)]
    # hydrant zone 13..23 cuts 13..20 out of the first segment
    assert first["no_parking_m"] == 7.0 and first["usable_m"] == 13.0
    # cars cover 8..12 and 12.5..16.5 -> 8 m, of which 13..16.5 is in the zone
    assert first["occupied_m"] == 8.0
    assert first["occupied_usable_m"] == 4.5 and first["occupied_no_parking_m"] == 3.5
    assert first["occupancy"] == pytest.approx(4.5 / 13, abs=1e-4)
    assert first["vehicles"] == 2
    assert first["track_ids"] == "3;7" and first["first_seen_s"] == "2.5;12.0"
    second = segs.loc[("left", 20.0)]
    assert second["no_parking_m"] == 3.0 and second["usable_m"] == 17.0
    assert second["occupancy"] == 0.0 and second["vehicles"] == 0
    right = segs.loc[("right", 20.0)]
    assert right["occupied_m"] == 5.0 and right["occupancy"] == 0.25


def test_build_segments_last_segment_is_shorter_and_empty_inputs_work():
    empty_cars = pd.DataFrame(columns=["track_id", "side", "s", "length_m", "first_seen_s"])
    empty_zones = pd.DataFrame(columns=["side", "s", "half_width_m"])
    segs = sg.build_segments(45.0, 20.0, empty_cars, empty_zones)
    assert len(segs) == 6 and segs["segment_m"].tolist()[-1] == 5.0
    assert (segs["occupancy"] == 0).all()


def test_fully_blocked_segment_has_no_occupancy():
    cars = pd.DataFrame(columns=["track_id", "side", "s", "length_m", "first_seen_s"])
    zone = pd.DataFrame({"side": ["right"], "s": [10.0], "half_width_m": [15.0]})
    segs = sg.build_segments(20.0, 20.0, cars, zone).set_index("side")
    assert segs.loc["right", "usable_m"] == 0 and pd.isna(segs.loc["right", "occupancy"])


def test_geojson_is_valid_and_in_lon_lat_order():
    line = sg.path_line(np.array([[0.0, 0.0], [40.0, 0.0]]))
    vehicles = pd.DataFrame({"track_id": [7], "class": ["car"],
                             "parked": pd.array([True], dtype="boolean"),
                             "first_seen_s": [12.0], "side": ["left"], "length_m": [np.nan],
                             "length_source": [None], "x": [10.0], "y": [3.0]})
    data = sg.geojson(line, segments_fixture(), vehicles, (49.0, 8.4), display_offset_m=4.0)
    text = json.dumps(data)  # must serialise: no numpy or pandas scalars left
    kinds = [f["geometry"]["type"] for f in data["features"]]
    assert kinds.count("LineString") == 4 and kinds.count("Point") == 1
    lon, lat = data["features"][-1]["geometry"]["coordinates"]
    assert lon == pytest.approx(8.4, abs=0.01) and lat == pytest.approx(49.0, abs=0.01)
    point = json.loads(text)["features"][-1]["properties"]
    assert point["parked"] is True and point["length_m"] is None
    left = data["features"][0]["geometry"]["coordinates"]
    assert left[0][1] > 49.0  # left segment drawn north of an eastbound path


def test_segments_stage_refuses_without_frozen_thresholds(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--stage",
                        "segments", "--frozen", str(tmp_path / "none.yaml"), "--out-dir",
                        str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "parked/moving thresholds" in r.stderr


def test_geojson_survives_a_tight_bend():
    # A hairpin: offsetting the inner side of this piece gives a multi-part line.
    t = np.linspace(0, np.pi, 30)
    path = np.vstack([np.column_stack([np.linspace(-20, 0, 10), np.full(10, -3.0)]),
                      np.column_stack([3 * np.sin(t), -3 * np.cos(t)]),
                      np.column_stack([np.linspace(0, -20, 10), np.full(10, 3.0)])])
    line = sg.path_line(path)
    empty_cars = pd.DataFrame(columns=["track_id", "side", "s", "length_m", "first_seen_s"])
    empty_zones = pd.DataFrame(columns=["side", "s", "half_width_m"])
    segs = sg.build_segments(line.length, 20.0, empty_cars, empty_zones)
    data = sg.geojson(line, segs, empty_cars.assign(x=[], y=[]), (49.0, 8.4), 4.0)
    assert all(f["geometry"]["type"] == "LineString" for f in data["features"])
    json.dumps(data)
