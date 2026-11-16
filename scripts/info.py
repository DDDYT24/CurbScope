"""Print a short summary of one KITTI drive and write it to results/info_<drive>.json.

Usage: python scripts/info.py --kitti-dir ~/data/kitti --drive 2011_09_26_drive_0019
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from curbscope import geo, io_kitti

# A tracklet whose labelled world position spreads less than this is "static".
# Only used to judge which drives have many parked cars; M3 uses my own labels instead.
STATIC_SPREAD_M = 1.0


def tracklet_stats(tracklets, T_w_imu, T_velo_imu) -> dict:
    counts = Counter(t.object_type for t in tracklets)
    static = Counter()
    for t in tracklets:
        world = np.array(
            [
                geo.apply_transform(geo.world_from_velo(T_w_imu[f], T_velo_imu), pose[:3])[0]
                for f, pose in zip(t.frames, t.poses, strict=True)
                if f < len(T_w_imu)
            ]
        )
        spread = np.ptp(world[:, :2], axis=0).max() if len(world) else np.inf
        if spread < STATIC_SPREAD_M:
            static[t.object_type] += 1
    return {"by_class": dict(counts), "static_by_class": dict(static)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-dir", required=True)
    parser.add_argument("--drive", required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    raw = io_kitti.open_drive(args.kitti_dir, args.drive)
    T_w_imu = io_kitti.poses(raw)
    info = {"drive": args.drive}
    info.update(io_kitti.drive_summary(io_kitti.timestamps_s(raw), T_w_imu[:, :3, 3]))
    first = raw.oxts[0].packet
    info["start_lat_lon"] = [first.lat, first.lon]

    xml = io_kitti.drive_dir(args.kitti_dir, args.drive) / "tracklet_labels.xml"
    if xml.exists():
        calib = io_kitti.calibration(raw)
        stats = tracklet_stats(io_kitti.load_tracklets(xml), T_w_imu, calib["T_velo_imu"])
        info["tracklets"] = stats
    else:
        info["tracklets"] = None

    print(f"drive        {info['drive']}")
    print(f"frames       {info['frames']}")
    print(f"duration     {info['duration_s']:.1f} s")
    print(f"distance     {info['distance_m']:.1f} m")
    print(f"start        {first.lat:.6f}, {first.lon:.6f}")
    if info["tracklets"] is None:
        print("tracklets    missing (tracklet_labels.xml not found)")
    else:
        for cls, n in sorted(info["tracklets"]["by_class"].items()):
            n_static = info["tracklets"]["static_by_class"].get(cls, 0)
            print(f"tracklets    {cls:<16}{n:>4}  static {n_static}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"info_{args.drive}.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
