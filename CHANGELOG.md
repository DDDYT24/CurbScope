# Changelog

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
