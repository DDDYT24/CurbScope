# Changelog

## M3 Parked or moving

- `curbscope/motion.py`: per-track features for an image rule (box shift over box height) and a
  world rule (spread around the median position, implied speed); scoring of both rules on the
  same labelled tracks with accuracy, precision, recall and confusion matrix.
- `apps/label_tracks.py` (`make label`): labelling page with box crops over time, a full frame
  and the world trajectory; writes `labels/track_labels_<drive>.csv`; never shows rule results.
- `scripts/tune.py` (`make tune`): accuracy over a threshold grid on dev drives; refuses the
  test drive.
- `scripts/evaluate.py` (`make evaluate SPLIT=dev|test`): uses `configs/frozen_thresholds.yaml`;
  the test drive is refused unless that file is committed and unchanged; its SHA-256 goes into
  `results/motion_eval.json`.

## M2 Georeference

- `curbscope/geo.py`: project Velodyne points into cam2, keep the central part of each box above
  the ground, take the nearest cluster's median as the position, convert to world and to
  latitude, longitude by inverting pykitti's Mercator exactly (pyproj).
- `curbscope/geo_eval.py`: per-frame Hungarian matching against KITTI Car/Van/Truck tracklets
  (4 m gate), errors to the label centre and to its footprint, by range.
- `make geo` writes `results/geo_<drive>.csv`, `results/geo_error_<drive>.json`,
  `results/geo_error.png` and a calibration check image per drive.

## M1 Detect and track

- `curbscope/detect_track.py`: YOLO11n (COCO weights) on every cam2 frame, ByteTrack IDs through
  Ultralytics `track(..., persist=True)`, classes car, motorcycle, bus, truck, fire hydrant and
  stop sign. A callback counts detections before tracking.
- `configs/default.yaml` (drive split, detection settings, demo GIF) and `configs/bytetrack.yaml`
  (Ultralytics 8.4.196 defaults, untuned).
- `make weights` downloads the weights and checks their SHA-256; `make track [DRIVE=<id>]` writes
  `results/tracks_<drive>.csv`, `results/tracks_summary.json` and a demo GIF under 5 MB.
- Ultralytics runs offline (no analytics, no automatic installs); PyTorch is the CPU build.

## M0 Scaffold and data loading

- Project packaging with pinned dependencies (`pyproject.toml`, `uv.lock`), ruff, pytest,
  Makefile and a CI workflow for lint and unit tests.
- `curbscope/io_kitti.py`: loads a KITTI raw drive through pykitti (cam2 images, Velodyne scans,
  OXTS poses, timestamps, calibration) and parses `tracklet_labels.xml`.
- `curbscope/geo.py`: rigid transform helpers (invert, apply, camera and Velodyne to world).
- `make info DRIVE=<id>`: frames, duration, distance driven, start point and tracklet counts,
  also written to `results/info_<drive>.json`.
- Tests run on a tiny fake drive and a small tracklet file, so CI needs no KITTI data.
- `docs/download_kitti.md`: the three proposed drives and how to download and unpack them.
