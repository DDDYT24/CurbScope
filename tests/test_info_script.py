"""Run scripts/info.py end to end on the fake drive from conftest.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import DRIVE

SCRIPT = Path(__file__).parents[1] / "scripts" / "info.py"
FIXTURES = Path(__file__).parent / "fixtures"


def run_info(kitti_dir: Path, out_dir: Path) -> str:
    cmd = [sys.executable, str(SCRIPT), "--kitti-dir", str(kitti_dir), "--drive", DRIVE,
           "--out-dir", str(out_dir)]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def test_info_without_tracklets(fake_kitti, tmp_path):
    stdout = run_info(fake_kitti, tmp_path / "results")
    assert "frames       3" in stdout
    assert "tracklets    missing" in stdout
    info = json.loads((tmp_path / "results" / f"info_{DRIVE}.json").read_text())
    assert info["frames"] == 3
    assert info["duration_s"] == pytest.approx(0.2)
    assert info["tracklets"] is None


def test_info_counts_static_tracklets(fake_kitti, tmp_path):
    sync = fake_kitti / "2011_09_26" / f"{DRIVE}_sync"
    # Fixture car: frames 3-4 do not exist in the 3-frame fake drive, so it has no world points
    # and must not count as static. Pedestrian: one frame, zero spread, counts as static.
    (sync / "tracklet_labels.xml").write_bytes((FIXTURES / "tracklets_small.xml").read_bytes())
    run_info(fake_kitti, tmp_path / "results")
    info = json.loads((tmp_path / "results" / f"info_{DRIVE}.json").read_text())
    assert info["tracklets"]["by_class"] == {"Car": 1, "Pedestrian": 1}
    assert info["tracklets"]["static_by_class"] == {"Pedestrian": 1}
