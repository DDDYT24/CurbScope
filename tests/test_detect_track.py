"""detect_track on fake detections: no model, no weights, no KITTI."""

from types import SimpleNamespace

import numpy as np
import pytest

from curbscope import detect_track as dt
from curbscope.config import all_drives, load_config

NAMES = {0: "person", 2: "car", 7: "truck", 10: "fire hydrant", 11: "stop sign"}


def fake_boxes(rows):
    """rows: (x1, y1, x2, y2, conf, cls, track_id); an empty list means no tracks."""
    if not rows:
        return SimpleNamespace(xyxy=np.zeros((0, 4)), conf=np.zeros(0), cls=np.zeros(0), id=None)
    a = np.array(rows, dtype=float)
    return SimpleNamespace(xyxy=a[:, :4], conf=a[:, 4], cls=a[:, 5], id=a[:, 6])


def test_class_ids_maps_names():
    assert dt.class_ids(NAMES, ["car", "stop sign"]) == [2, 11]


def test_class_ids_rejects_unknown():
    with pytest.raises(ValueError, match="bicycle"):
        dt.class_ids(NAMES, ["car", "bicycle"])


def test_boxes_to_rows_columns_and_values():
    rows = dt.boxes_to_rows(5, 0.5, fake_boxes([(1, 2, 3, 4, 0.9, 2, 7)]), NAMES)
    assert list(rows[0]) == dt.COLUMNS
    assert rows[0]["track_id"] == 7 and rows[0]["class"] == "car" and rows[0]["frame"] == 5


def test_boxes_without_ids_give_no_rows():
    assert dt.boxes_to_rows(0, 0.0, fake_boxes([]), NAMES) == []


def test_track_frames_passes_bgr_and_collects_rows():
    seen = []

    def track_fn(image_bgr):
        seen.append(image_bgr[0, 0].tolist())
        return fake_boxes([(0, 0, 10, 10, 0.8, 2, 1)])

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255  # pure red in RGB
    df = dt.track_frames([(0, 0.0, image), (1, 0.1, image)], track_fn, NAMES)
    assert seen[0] == [0, 0, 255]  # red arrives as BGR
    assert list(df.columns) == dt.COLUMNS
    assert df["frame"].tolist() == [0, 1]


def test_track_class_uses_majority_and_summary_counts():
    def boxes_for(frame):
        cls = 7 if frame == 0 else 2  # track 1: truck once, car twice -> car
        return fake_boxes([(0, 0, 5, 5, 0.9, cls, 1), (0, 0, 5, 5, 0.5, 10, 2)])

    frames = [(i, i / 10, np.zeros((2, 2, 3), np.uint8)) for i in range(3)]
    calls = iter(range(3))
    df = dt.track_frames(frames, lambda _img: boxes_for(next(calls)), NAMES)
    assert dt.track_class(df).to_dict() == {1: "car", 2: "fire hydrant"}

    summary = dt.summarize(df, n_frames=3, raw_detections=8, seconds=1.5)
    assert summary["tracks"] == 2
    assert summary["tracks_by_class"] == {"car": 1, "fire hydrant": 1}
    assert summary["detections_tracked"] == 6
    assert summary["detections_without_track"] == 2
    assert summary["cpu_seconds_per_frame"] == 0.5


def test_check_weights(tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"abc")
    good = dt.sha256(w)
    assert dt.check_weights(w, good) == good
    with pytest.raises(ValueError):
        dt.check_weights(w, "0" * 64)
    with pytest.raises(FileNotFoundError, match="make weights"):
        dt.check_weights(tmp_path / "missing.pt", good)


def test_draw_boxes_does_not_modify_input():
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    rows = [{"track_id": 3, "class": "car", "x1": 5, "y1": 5, "x2": 40, "y2": 30}]
    out = dt.draw_boxes(image, rows)
    assert image.sum() == 0 and out.sum() > 0


def test_gif_is_small_and_size_limit_enforced(tmp_path, monkeypatch):
    frames = [np.full((375, 1242, 3), i * 20, dtype=np.uint8) for i in range(5)]
    n = dt.write_gif(frames, tmp_path / "a.gif", width=620, fps=5)
    assert 0 < n < dt.GIF_MAX_BYTES
    monkeypatch.setattr(dt, "GIF_MAX_BYTES", 10)
    with pytest.raises(ValueError, match="5 MB"):
        dt.write_gif(frames, tmp_path / "b.gif", width=620, fps=5)


def test_default_config_is_consistent():
    cfg = load_config()
    assert set(cfg["drives"]["dev"]).isdisjoint(cfg["drives"]["test"])
    assert cfg["demo_gif"]["drive"] in all_drives(cfg)
    assert len(cfg["detect"]["weights_sha256"]) == 64
