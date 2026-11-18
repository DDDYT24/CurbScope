"""Parked/moving features, rules, scoring and the scripts' refusal rules, on synthetic tracks."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curbscope import labeling, motion

ROOT = Path(__file__).parents[1]
CFG = {"min_frames_rule_a": 3, "min_known_rule_b": 3}


def synthetic_drive():
    """Track 1: parked (box slides across the image, world position fixed).
    Track 2: moving at 5 m/s. Track 3: 2 frames only (no rule can decide)."""
    boxes, located = [], []
    for f in range(10):
        t = f / 10
        boxes.append({"frame": f, "timestamp": t, "track_id": 1, "class": "car", "confidence": .9,
                      "x1": 100 + 40 * f, "y1": 100, "x2": 200 + 40 * f, "y2": 150})
        located.append({"frame": f, "timestamp": t, "track_id": 1, "status": "known",
                        "x": 10.0 + 0.05 * (-1) ** f, "y": 3.0})
        boxes.append({"frame": f, "timestamp": t, "track_id": 2, "class": "car", "confidence": .9,
                      "x1": 600, "y1": 100, "x2": 700, "y2": 150})
        located.append({"frame": f, "timestamp": t, "track_id": 2, "status": "known",
                        "x": 20.0 + 5 * t, "y": -3.0})
    for f in range(2):
        boxes.append({"frame": f, "timestamp": f / 10, "track_id": 3, "class": "truck",
                      "confidence": .5, "x1": 0, "y1": 0, "x2": 10, "y2": 10})
        located.append({"frame": f, "timestamp": f / 10, "track_id": 3, "status": "unknown",
                        "x": np.nan, "y": np.nan})
    return pd.DataFrame(boxes), pd.DataFrame(located)


@pytest.fixture
def features():
    return motion.track_features(*synthetic_drive(), CFG).assign(drive="d1")


def test_features(features):
    f = features.set_index("track_id")
    assert f.loc[1, "box_shift_over_height"] == pytest.approx(360 / 50)  # 9 steps of 40 px
    assert f.loc[1, "spread_m"] == pytest.approx(0.05)
    assert f.loc[2, "box_shift_over_height"] == 0.0
    assert f.loc[2, "speed_mps"] == pytest.approx(5.0, rel=0.05)
    assert np.isnan(f.loc[3, "box_shift_over_height"]) and np.isnan(f.loc[3, "spread_m"])
    assert f.loc[3, "class"] == "truck" and f.loc[1, "first_seen_s"] == 0.0


def test_rules_show_the_image_rule_failure(features):
    a = motion.rule_a(features, 1.0).tolist()
    b = motion.rule_b(features, 0.5, 1.0).tolist()
    assert a[:2] == [False, True] and pd.isna(a[2])   # image rule gets both wrong
    assert b[:2] == [True, False] and pd.isna(b[2])   # world rule gets both right


def test_rule_b_uses_spread_alone_without_speed():
    f = pd.DataFrame({"spread_m": [0.2, 0.2], "speed_mps": [np.nan, 9.0]})
    assert motion.rule_b(f, 0.5, 1.0).tolist() == [True, False]


def test_score_compares_on_tracks_every_rule_decides(features):
    labels = pd.DataFrame({"drive": "d1", "track_id": [1, 2, 3], "label": ["parked", "moving",
                                                                           "parked"]})
    preds = {"a": motion.rule_a(features, 1.0), "b": motion.rule_b(features, 0.5, 1.0)}
    s = motion.score(labels, preds, features)
    assert s["compared_tracks"] == 2 and s["not_decided_by_every_rule"] == 1
    assert s["rules"]["b"]["accuracy"] == 1.0 and s["rules"]["a"]["accuracy"] == 0.0
    assert s["rules"]["a"]["confusion"] == {"parked_as_parked": 0, "moving_as_parked": 1,
                                            "parked_as_moving": 1, "moving_as_moving": 0}


def test_score_keeps_drives_apart(features):
    other = features.assign(drive="d2")
    both = pd.concat([features, other], ignore_index=True)
    labels = pd.DataFrame({"drive": ["d1", "d2"], "track_id": [1, 1], "label": ["parked",
                                                                                "unsure"]})
    s = motion.score(labels, {"b": motion.rule_b(both, 0.5, 1.0)}, both)
    assert s["labelled_tracks"] == 2 and s["unsure"] == 1 and s["compared_tracks"] == 1


def test_load_frozen_validates(tmp_path):
    good = tmp_path / "frozen.yaml"
    good.write_text("rule_a: {max_shift_over_height: 2}\n"
                    "rule_b: {max_spread_m: 1.0, max_speed_mps: 1.0}\n")
    frozen, sha = motion.load_frozen(good)
    assert frozen["rule_b"]["max_spread_m"] == 1.0 and len(sha) == 64
    bad = tmp_path / "bad.yaml"
    bad.write_text("rule_a: {max_shift_over_height: two}\n")
    with pytest.raises(ValueError, match="rule_a.max_shift_over_height"):
        motion.load_frozen(bad)
    with pytest.raises(FileNotFoundError, match="make tune"):
        motion.load_frozen(tmp_path / "missing.yaml")


def test_load_split_requires_labels(tmp_path, features):
    features.drop(columns="drive").to_csv(tmp_path / "motion_features_d1.csv", index=False)
    with pytest.raises(FileNotFoundError, match="make label"):
        motion.load_split(["d1"], tmp_path, tmp_path)
    labeling.save_label(tmp_path / "track_labels_d1.csv", 1, "parked")
    f, labels = motion.load_split(["d1"], tmp_path, tmp_path)
    assert set(f["drive"]) == {"d1"} and labels["track_id"].tolist() == [1]


def test_load_labels_rejects_unknown_and_duplicates(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("track_id,label,note\n1,parkd,\n")
    with pytest.raises(ValueError, match="unknown labels"):
        motion.load_labels(p)
    p.write_text("track_id,label,note\n1,parked,\n1,moving,\n")
    with pytest.raises(ValueError, match="twice"):
        motion.load_labels(p)


def run_script(name, *args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_tune_refuses_test_split():
    r = run_script("tune.py", "--split", "test")
    assert r.returncode != 0 and "never looks at the test drive" in r.stderr


def test_evaluate_refuses_without_frozen_file(tmp_path):
    r = run_script("evaluate.py", "--split", "dev", "--frozen", str(tmp_path / "none.yaml"))
    assert r.returncode != 0 and "make tune" in r.stderr


def test_evaluate_refuses_test_with_uncommitted_frozen_file(tmp_path):
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text("rule_a: {max_shift_over_height: 2}\n"
                      "rule_b: {max_spread_m: 1.0, max_speed_mps: 1.0}\n")
    r = run_script("evaluate.py", "--split", "test", "--frozen", str(frozen))
    assert r.returncode != 0 and "must be committed" in r.stderr


def test_parked_by_track_gives_plain_python_values(features):
    frozen = {"rule_a": {"max_shift_over_height": 1.0},
              "rule_b": {"max_spread_m": 0.5, "max_speed_mps": 1.0}}
    status = motion.parked_by_track(features, frozen)
    assert status[1] is True and status[2] is False and status[3] is None


def test_committed_unchanged_detects_an_edit_after_commit(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    frozen = Path("frozen.yaml")
    frozen.write_text("rule_a: {max_shift_over_height: 2}\n")
    assert not evaluate.committed_unchanged(frozen)          # not tracked yet
    subprocess.run(git + ["add", "frozen.yaml"], check=True)
    subprocess.run(git + ["commit", "-qm", "freeze"], check=True)
    assert evaluate.committed_unchanged(frozen)
    frozen.write_text("rule_a: {max_shift_over_height: 3}\n")
    assert not evaluate.committed_unchanged(frozen)          # edited after the commit


def test_tune_without_inputs_explains_instead_of_a_traceback(tmp_path):
    r = run_script("tune.py", "--split", "dev", "--results-dir", str(tmp_path))
    assert r.returncode != 0 and "make motion" in r.stderr and "Traceback" not in r.stderr
