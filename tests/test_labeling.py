import numpy as np
import pytest

from curbscope import labeling


def test_save_label_adds_replaces_and_keeps_notes(tmp_path):
    path = tmp_path / "labels" / "track_labels_d.csv"
    labeling.save_label(path, 5, "parked")
    labeling.save_label(path, 2, "unsure", "  ID switch with 9 ")
    labels = labeling.save_label(path, 5, "moving")
    assert labels["track_id"].tolist() == [2, 5]
    again = labeling.read_labels(path)
    assert again.set_index("track_id")["label"].to_dict() == {2: "unsure", 5: "moving"}
    assert again.set_index("track_id").loc[2, "note"] == "ID switch with 9"
    assert not path.with_suffix(".tmp").exists()


def test_save_label_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        labeling.save_label(tmp_path / "l.csv", 1, "stopped")


def test_read_labels_missing_file_is_empty(tmp_path):
    assert labeling.read_labels(tmp_path / "none.csv").empty


def test_pick_frames():
    assert labeling.pick_frames(np.array([3, 1, 2]), 8) == [1, 2, 3]
    picked = labeling.pick_frames(np.arange(100), 5)
    assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5


def test_crop_box_pads_and_clips():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    # rows 34..66 and columns 20..180, both ends included
    assert labeling.crop_box(image, (50, 40, 150, 60), pad=0.3).shape == (33, 161, 3)
    assert labeling.crop_box(image, (0, 0, 20, 20), pad=0.5).shape == (31, 31, 3)
