"""A tiny fake KITTI drive on disk, so the pykitti loading path runs in CI without real data."""

from pathlib import Path

import pytest

DATE = "2011_09_26"
DRIVE = f"{DATE}_drive_0001"
IDENTITY_R = "1 0 0 0 1 0 0 0 1"
P = "700 0 600 0 0 700 180 0 0 0 1 0"


def write_fake_drive(root: Path, n_frames: int = 3, step_lat_deg: float = 1e-5) -> Path:
    date_dir = root / DATE
    sync = date_dir / f"{DRIVE}_sync"
    (sync / "oxts" / "data").mkdir(parents=True)
    (sync / "velodyne_points" / "data").mkdir(parents=True)

    cam = [f"P_rect_0{i}: {P}" for i in range(4)] + [f"R_rect_0{i}: {IDENTITY_R}" for i in range(4)]
    (date_dir / "calib_cam_to_cam.txt").write_text("\n".join(cam) + "\n")
    (date_dir / "calib_velo_to_cam.txt").write_text(f"R: {IDENTITY_R}\nT: 0 0 -0.1\n")
    (date_dir / "calib_imu_to_velo.txt").write_text(f"R: {IDENTITY_R}\nT: -0.8 0.3 -0.8\n")

    stamps = []
    for i in range(n_frames):
        # lat lon alt roll pitch yaw, then 19 rates/accelerations, then 5 integer status fields
        values = [49.0 + i * step_lat_deg, 8.4, 110.0, 0.0, 0.0, 0.0] + [0.0] * 19
        line = " ".join(f"{v:.10f}" for v in values) + " 4 10 4 4 0"
        (sync / "oxts" / "data" / f"{i:010d}.txt").write_text(line + "\n")
        stamps.append(f"2011-09-26 13:02:25.{i:01d}00000000")
    for sub in ("oxts", "velodyne_points"):
        (sync / sub / "timestamps.txt").write_text("\n".join(stamps) + "\n")
    return root


@pytest.fixture
def fake_kitti(tmp_path) -> Path:
    return write_fake_drive(tmp_path)
