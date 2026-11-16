"""Run the pipeline stages on KITTI drives.

Stages: track (M1, YOLO + ByteTrack), geo (M2, LiDAR position, lat/lon, error vs labels).
Usage: python scripts/run_pipeline.py --stage track --kitti-dir ~/data/kitti [--drive <id>]
Without --drive, every dev and test drive in the config is processed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from curbscope import detect_track, geo, geo_eval, io_kitti
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


def stage_track(cfg: dict, kitti_dir: str, drives: list[str], out_dir: Path) -> None:
    import ultralytics

    weights_hash = detect_track.check_weights(
        cfg["detect"]["weights"], cfg["detect"]["weights_sha256"]
    )
    summary_path = out_dir / "tracks_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"drives": {}}
    summary["settings"] = {
        "ultralytics": ultralytics.__version__,
        "weights": Path(cfg["detect"]["weights"]).name,
        "weights_sha256": weights_hash,
        **{k: cfg["detect"][k] for k in ("imgsz", "conf", "classes", "tracker")},
    }
    for drive in drives:
        print(f"tracking {drive} ...", flush=True)
        result = run_track(cfg, kitti_dir, drive, out_dir)
        summary["drives"][drive] = result
        print(f"  {result['tracks']} tracks {result['tracks_by_class']}, "
              f"{result['cpu_seconds_per_frame']} CPU s/frame", flush=True)
    summary["drives"] = dict(sorted(summary["drives"].items()))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def locate_tracks(raw, calib: dict, tracks: pd.DataFrame, T_w_cam2: np.ndarray,
                  cfg_geo: dict) -> pd.DataFrame:
    """Add status, n_points, cam_x/y/z and world x/y/z to every tracked box."""
    K = calib["P_rect_20"][:3, :3]
    rows = []
    for frame, group in tracks.groupby("frame"):
        cam, uv = geo.frame_points(io_kitti.velo_scan(raw, int(frame)), calib["T_cam2_velo"],
                                   K, cfg_geo)
        for r in group.to_dict("records"):
            box = (r["x1"], r["y1"], r["x2"], r["y2"])
            points = geo.box_points(cam, uv, box, cfg_geo["box_keep_fraction"])
            pos, n = geo.locate(points, cfg_geo)
            row = r | {"status": "unknown" if pos is None else "known", "n_points": n}
            if pos is not None:
                world = geo.apply_transform(T_w_cam2[int(frame)], pos)[0]
                row |= {"cam_x": pos[0], "cam_y": pos[1], "cam_z": pos[2],
                        "x": world[0], "y": world[1], "z": world[2],
                        "range_m": float(np.hypot(pos[0], pos[2]))}
            rows.append(row)
    return pd.DataFrame(rows)


def labels_in_view(labels: pd.DataFrame, tracklets, calib: dict, image_shape) -> pd.DataFrame:
    """Keep labels whose box centre projects inside the camera image."""
    K = calib["P_rect_20"][:3, :3]
    h, w = image_shape[:2]
    keep = []
    for r in labels.itertuples(index=False):
        t = tracklets[r.tracklet]
        pose = t.poses[r.frame - t.first_frame]
        centre = pose[:3] + np.array([0, 0, t.h / 2])
        cam = geo.apply_transform(calib["T_cam2_velo"], centre)
        u, v = geo.project(cam, K)[0]
        keep.append(bool(cam[0, 2] > 0.5 and 0 <= u < w and 0 <= v < h))
    return labels[np.array(keep, dtype=bool)] if len(labels) else labels


def run_geo(cfg: dict, kitti_dir: str, drive: str, split: str, out_dir: Path):
    tracks_path = out_dir / f"tracks_{drive}.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(f"{tracks_path} not found; run `make track` first")
    raw = io_kitti.open_drive(kitti_dir, drive)
    calib = io_kitti.calibration(raw)
    T_w_imu = io_kitti.poses(raw)
    T_w_cam2 = np.stack([geo.world_from_cam2(T, calib["T_cam2_imu"]) for T in T_w_imu])
    T_w_velo = np.stack([geo.world_from_velo(T, calib["T_velo_imu"]) for T in T_w_imu])
    first = raw.oxts[0].packet

    # Lat/lon: check the inverse Mercator against every OXTS fix, then convert.
    fixes = np.array([[o.packet.lat, o.packet.lon] for o in raw.oxts])
    back = geo.latlon_to_world(fixes, first.lat, first.lon)
    latlon_check_m = float(np.max(np.linalg.norm(back - T_w_imu[:, :2, 3], axis=1)))

    located = locate_tracks(raw, calib, pd.read_csv(tracks_path), T_w_cam2, cfg["geo"])
    known = located["status"] == "known"
    latlon = geo.world_to_latlon(located.loc[known, ["x", "y"]].to_numpy(), first.lat,
                                 first.lon)
    located.loc[known, "lat"] = latlon[:, 0]
    located.loc[known, "lon"] = latlon[:, 1]
    columns = ["frame", "timestamp", "track_id", "class", "status", "n_points", "range_m",
               "x", "y", "z", "lat", "lon", "cam_x", "cam_y", "cam_z"]
    # Metres at 3 decimals (1 mm); lat/lon at 7 decimals (about 1 cm).
    out = located[columns].copy()
    for col in ("lat", "lon"):
        out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.7f}")
    out.to_csv(out_dir / f"geo_{drive}.csv", index=False, float_format="%.3f")

    ev = cfg["geo_eval"]
    xml = io_kitti.drive_dir(kitti_dir, drive) / "tracklet_labels.xml"
    tracklets = io_kitti.load_tracklets(xml)
    labels = geo_eval.tracklets_by_frame(tracklets, T_w_velo, ev["tracklet_types"])
    image_shape = io_kitti.cam2_image(raw, 0).shape
    labels = labels_in_view(labels, tracklets, calib, image_shape)
    vehicles = located[located["class"].isin(ev["detection_classes"])]
    pairs = geo_eval.match_drive(vehicles[vehicles["status"] == "known"], labels,
                                 ev["match_max_m"])

    n_known = int((vehicles["status"] == "known").sum())
    result = {
        "drive": drive,
        "split": split,
        "vehicle_boxes": int(len(vehicles)),
        "vehicle_boxes_unknown": int((vehicles["status"] == "unknown").sum()),
        "vehicle_boxes_known": n_known,
        "label_instances_in_view": int(len(labels)),
        "matched_pairs": int(len(pairs)),
        "known_boxes_without_label": n_known - int(len(pairs)),
        "labels_without_box": int(len(labels)) - int(len(pairs)),
        "error_by_range": geo_eval.error_by_range(pairs, ev["range_bins_m"]),
        "all_classes_unknown": int((~known).sum()),
        "all_classes_boxes": int(len(located)),
        "latlon_roundtrip_max_m": round(latlon_check_m, 6),
        "settings": {"geo": cfg["geo"], "geo_eval": ev},
    }
    (out_dir / f"geo_error_{drive}.json").write_text(json.dumps(result, indent=2) + "\n")

    # Visual check on the frame with the most labelled vehicles in view.
    from curbscope import geo_plots

    frame = int(labels["frame"].value_counts().idxmax())
    in_frame = labels[labels["frame"] == frame]
    corners = []
    for r in in_frame.itertuples(index=False):
        t = tracklets[r.tracklet]
        corners.append(geo_eval.box_corners_velo(t.poses[frame - t.first_frame], t.h, t.w, t.l))
    geo_plots.plot_check(io_kitti.cam2_image(raw, frame), io_kitti.velo_scan(raw, frame), calib,
                         located[located["frame"] == frame], corners,
                         out_dir / f"geo_check_{drive}.png", f"{drive}, frame {frame}")
    return result, pairs.assign(split=split, drive=drive)


def stage_geo(cfg: dict, kitti_dir: str, drives: list[str], out_dir: Path) -> None:
    from curbscope import geo_plots

    all_pairs = []
    for drive in drives:
        split = "test" if drive in cfg["drives"]["test"] else "dev"
        print(f"locating {drive} ({split}) ...", flush=True)
        result, pairs = run_geo(cfg, kitti_dir, drive, split, out_dir)
        all_pairs.append(pairs)
        err = result["error_by_range"]["all"]
        print(f"  {result['matched_pairs']} matched pairs, center median "
              f"{err['center']['median']} m, footprint median {err['footprint']['median']} m, "
              f"{result['vehicle_boxes_unknown']}/{result['vehicle_boxes']} boxes unknown",
              flush=True)
    if len(drives) == len(all_drives(cfg)):
        geo_plots.plot_errors(pd.concat(all_pairs), out_dir / "geo_error.png",
                              cfg["geo_eval"]["range_bins_m"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["track", "geo"], required=True)
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--kitti-dir")
    parser.add_argument("--drive", help="one drive id; default: all drives in the config")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kitti_dir = args.kitti_dir or cfg["kitti_dir"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drives = [args.drive] if args.drive else all_drives(cfg)
    {"track": stage_track, "geo": stage_geo}[args.stage](cfg, kitti_dir, drives, out_dir)


if __name__ == "__main__":
    main()
