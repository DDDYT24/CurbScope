"""Run the pipeline stages on KITTI drives. M1: detect and track.

Usage: python scripts/run_pipeline.py --kitti-dir ~/data/kitti [--drive 2011_09_26_drive_0022]
Without --drive, every dev and test drive in the config is processed.
"""

import argparse
import json
import time
from pathlib import Path

import ultralytics

from curbscope import detect_track, io_kitti
from curbscope.config import DEFAULT, all_drives, load_config


def run_track(cfg: dict, kitti_dir: str, drive: str, out_dir: Path) -> dict:
    raw = io_kitti.open_drive(kitti_dir, drive)
    t_s = io_kitti.timestamps_s(raw)
    track_fn, names, counter = detect_track.make_yolo_tracker(cfg["detect"])

    gif_cfg = cfg["demo_gif"]
    gif_frames = []
    wanted = set(range(gif_cfg["first_frame"], gif_cfg["last_frame"] + 1, gif_cfg["step"]))

    def keep_for_gif(frame, image_rgb, rows):
        if drive == gif_cfg["drive"] and frame in wanted:
            gif_frames.append(detect_track.draw_boxes(image_rgb, rows))

    frames = ((i, float(t_s[i]), io_kitti.cam2_image(raw, i)) for i in range(len(t_s)))
    start = time.process_time()
    tracks = detect_track.track_frames(frames, track_fn, names, on_frame=keep_for_gif)
    seconds = time.process_time() - start

    tracks.to_csv(out_dir / f"tracks_{drive}.csv", index=False, float_format="%.3f")
    summary = detect_track.summarize(tracks, len(t_s), counter["raw"], seconds)
    if gif_frames:
        fps = 10 / gif_cfg["step"]
        path = out_dir / f"demo_{drive}.gif"
        summary["gif_bytes"] = detect_track.write_gif(gif_frames, path, gif_cfg["width"], fps)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--kitti-dir")
    parser.add_argument("--drive", help="one drive id; default: all drives in the config")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kitti_dir = args.kitti_dir or cfg["kitti_dir"]
    weights_hash = detect_track.check_weights(
        cfg["detect"]["weights"], cfg["detect"]["weights_sha256"]
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "tracks_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"drives": {}}
    summary["settings"] = {
        "ultralytics": ultralytics.__version__,
        "weights": Path(cfg["detect"]["weights"]).name,
        "weights_sha256": weights_hash,
        **{k: cfg["detect"][k] for k in ("imgsz", "conf", "classes", "tracker")},
    }

    for drive in [args.drive] if args.drive else all_drives(cfg):
        print(f"tracking {drive} ...", flush=True)
        result = run_track(cfg, kitti_dir, drive, out_dir)
        summary["drives"][drive] = result
        print(f"  {result['tracks']} tracks {result['tracks_by_class']}, "
              f"{result['cpu_seconds_per_frame']} CPU s/frame", flush=True)

    summary["drives"] = dict(sorted(summary["drives"].items()))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
