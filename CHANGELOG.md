# Changelog

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
