"""Load one KITTI raw drive (synced+rectified) and its tracklet labels.

Expected layout under kitti_dir (the layout of the official zip files):

    <kitti_dir>/2011_09_26/calib_cam_to_cam.txt, calib_imu_to_velo.txt, calib_velo_to_cam.txt
    <kitti_dir>/2011_09_26/2011_09_26_drive_0019_sync/{image_02,velodyne_points,oxts}/...
    <kitti_dir>/2011_09_26/2011_09_26_drive_0019_sync/tracklet_labels.xml
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pykitti

DRIVE_ID = re.compile(r"^(\d{4}_\d{2}_\d{2})_drive_(\d{4})$")


def parse_drive_id(drive_id: str) -> tuple[str, str]:
    """'2011_09_26_drive_0019' -> ('2011_09_26', '0019')."""
    match = DRIVE_ID.match(drive_id)
    if match is None:
        raise ValueError(f"drive id must look like 2011_09_26_drive_0019, got {drive_id!r}")
    return match.group(1), match.group(2)


def drive_dir(kitti_dir: str | Path, drive_id: str) -> Path:
    date, _ = parse_drive_id(drive_id)
    return Path(kitti_dir).expanduser() / date / f"{drive_id}_sync"


def open_drive(kitti_dir: str | Path, drive_id: str, frames=None) -> pykitti.raw:
    """Open a drive with pykitti. Images and scans are read lazily, one frame at a time."""
    date, number = parse_drive_id(drive_id)
    folder = drive_dir(kitti_dir, drive_id)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; see docs/download_kitti.md")
    return pykitti.raw(str(Path(kitti_dir).expanduser()), date, number, frames=frames)


def poses(raw: pykitti.raw) -> np.ndarray:
    """(N, 4, 4) T_w_imu per frame. World is pykitti's local metric frame at the first frame."""
    return np.stack([packet.T_w_imu for packet in raw.oxts])


def timestamps_s(raw: pykitti.raw) -> np.ndarray:
    """Seconds since the first frame, from the Velodyne-synced timestamps."""
    t0 = raw.timestamps[0]
    return np.array([(t - t0).total_seconds() for t in raw.timestamps])


def calibration(raw: pykitti.raw) -> dict[str, np.ndarray]:
    """The three matrices later milestones need, plus the Velodyne to IMU link."""
    calib = raw.calib
    return {
        "T_cam2_velo": calib.T_cam2_velo,
        "P_rect_20": calib.P_rect_20,
        "T_cam2_imu": calib.T_cam2_imu,
        "T_velo_imu": calib.T_velo_imu,
    }


def cam2_image(raw: pykitti.raw, index: int) -> np.ndarray:
    """Rectified left colour image as an (H, W, 3) uint8 RGB array."""
    return np.asarray(raw.get_cam2(index))


def velo_scan(raw: pykitti.raw, index: int) -> np.ndarray:
    """(N, 4) Velodyne points: x, y, z in metres (Velodyne frame) and reflectance."""
    return raw.get_velo(index)


@dataclass
class Tracklet:
    """One labelled object.

    poses rows are tx, ty, tz, rx, ry, rz in the Velodyne frame, one row per frame from
    first_frame on; (tx, ty, tz) is the bottom centre of the box.
    """

    object_type: str
    h: float
    w: float
    l: float  # noqa: E741  (KITTI's own name for box length)
    first_frame: int
    poses: np.ndarray

    @property
    def frames(self) -> np.ndarray:
        return np.arange(self.first_frame, self.first_frame + len(self.poses))


def load_tracklets(xml_path: str | Path) -> list[Tracklet]:
    """Parse tracklet_labels.xml (boost serialization format)."""
    root = ET.parse(xml_path).getroot()
    tracklets = []
    for item in root.find("tracklets").findall("item"):
        pose_rows = [
            [float(pose.find(key).text) for key in ("tx", "ty", "tz", "rx", "ry", "rz")]
            for pose in item.find("poses").findall("item")
        ]
        tracklets.append(
            Tracklet(
                object_type=item.find("objectType").text,
                h=float(item.find("h").text),
                w=float(item.find("w").text),
                l=float(item.find("l").text),
                first_frame=int(item.find("first_frame").text),
                poses=np.array(pose_rows, dtype=float).reshape(-1, 6),
            )
        )
    return tracklets


def drive_summary(t_s: np.ndarray, positions: np.ndarray) -> dict:
    """Frames, duration and horizontal distance driven from per-frame positions (N, 3)."""
    steps = np.diff(np.asarray(positions)[:, :2], axis=0)
    return {
        "frames": int(len(t_s)),
        "duration_s": float(t_s[-1] - t_s[0]) if len(t_s) else 0.0,
        "distance_m": float(np.linalg.norm(steps, axis=1).sum()),
    }
