# Architecture

## Data flow

```
KITTI drive ──► M0 io_kitti ──► M1 detect_track ──► results/tracks_<drive>.csv
(camera, LiDAR,                    (YOLO11n + ByteTrack)
 GPS/IMU, calib,                                 │
 tracklets)                                      ▼
                                   M2 geo, geo_eval ──► results/geo_<drive>.csv, geo_error_*.json
                                                 │
                       labels/track_labels_*.csv ▼
            configs/frozen_thresholds.yaml ─► M3 motion ──► motion_features_*.csv, motion_eval.json
                                                 │
            configs/rules_edmonton.yaml ───► M4 segments ──► segments_<drive>.csv, .geojson
                                                 │
 data/signs/ + labels/signs_groundtruth.csv ─► M5 ocr/* ──► ocr_benchmark.json, ocr_outputs.csv
                                                 │
                                   M6 report, apps/dashboard ──► results/report.md, README results
```

Each stage reads the previous stage's files in `results/` and writes its own, so any stage can be
rerun alone and every number in the report can be traced to the file and the script that wrote it.
Stages are run from `scripts/run_pipeline.py --stage <name>` (or the matching `make` target);
evaluation, tuning, OCR and the report have their own scripts.

## Modules

| Module | Role |
| --- | --- |
| `curbscope/io_kitti.py` | Open a KITTI raw drive with pykitti; poses, timestamps, calibration, images, scans; tracklet XML |
| `curbscope/geo.py` | Rigid transforms (`T_<to>_<from>`), LiDAR-to-image projection, object points in a box, world to lat/lon |
| `curbscope/detect_track.py` | YOLO + ByteTrack wrapper, tracks table, summary, demo GIF |
| `curbscope/geo_eval.py`, `geo_plots.py` | Matching located vehicles to KITTI labels, errors by range, figures |
| `curbscope/motion.py`, `labeling.py` | Per-track features, the two parked/moving rules, scoring, label files |
| `curbscope/segments.py` | Placing objects on our path, interval arithmetic, curb segments, GeoJSON |
| `curbscope/ocr/` | Text line detection, three recognisers, the sign rule parser |
| `curbscope/report.py` | Report and README results from `results/*.json` |
| `apps/label_tracks.py`, `apps/dashboard.py` | Streamlit labelling page and dashboard |
| `configs/` | All parameters, the drive split, the Edmonton rules; the owner's frozen thresholds |

Unit tests in `tests/` use synthetic data (a 3-frame fake KITTI drive, fake detections, synthetic
geometry), so CI never needs KITTI, model weights or OCR models.

## Decisions

### KITTI raw as the main dataset (2026-11-15)
- Context: the pipeline needs camera, LiDAR and GPS/IMU together, plus ground truth to measure
  position error.
- Options: KITTI raw (all sensors, calibration and 3D tracklets; non-commercial licence, Germany);
  own dashcam video (Edmonton, but no depth and no ground truth); Toronto-3D (LiDAR only).
- Decision: KITTI raw, three Residential drives with the most labelled cars, split by drive into
  two dev drives and one test drive whose routes do not overlap.
- Risks: German streets with Edmonton rules; results are a demonstration. Small sample.
- Reference: KITTI devkit and pykitti (MIT) for formats and transforms.

### Parked or moving in world coordinates, with an image rule as baseline (2026-11-17)
- Context: from a moving car, parked cars move in the image.
- Options: image displacement only (simple, fooled by ego motion); world position from LiDAR and
  pose (needs LiDAR points on the car); optical flow with ego-motion compensation (more parts).
- Decision: compute both an image rule and a world rule and let hand labels decide which works.
- Risks: LiDAR sees different sides of a car as we pass, so a parked car's world position spreads
  by up to about a metre; distant cars have no LiDAR position and cannot be decided.

### Thresholds chosen on dev drives and frozen before the test drive (2026-11-17)
- Context: thresholds chosen on the same data they are scored on give inflated numbers.
- Options: one split with cross-validation; dev/test split by drive with a committed threshold file.
- Decision: split by drive; `evaluate.py` refuses the test drive unless
  `configs/frozen_thresholds.yaml` is committed and unchanged, and stores its SHA-256.
- Risks: the test drive has few moving cars; accuracy is read with the confusion matrix.

### Position from the nearest LiDAR cluster in the box centre (2026-11-16)
- Context: a 2D box needs a depth.
- Options: box bottom on a flat ground plane (no LiDAR, wrong on slopes and for occlusion); median
  of all points in the box (background leaks in); nearest cluster in the box centre.
- Decision: central 60% of the box, ground points dropped, nearest cluster with at least 5
  points, its median; fewer than 10 points means unknown. Parameters were fixed before results.
- Risks: the cluster can belong to an occluder; the footprint error shows how often.

### Same crops for every OCR engine, one crop per line (2026-11-17)
- Context: a fair comparison of recognisers.
- Options: each engine detects and reads on its own (differences mix detection and recognition);
  one detector, same crops for all.
- Decision: EasyOCR's detector once; one crop per text line, because word crops dropped the dash
  in `9 AM - 6 PM` in a smoke test and every engine then lost the time range.
- Risks: a detector miss is a miss for all three engines; it is not attributed to any one.

### CPU only, pinned and two-week-old dependencies (2026-11-15, extended 2026-11-17)
- Context: the pipeline must run on a laptop and give the same numbers on every run.
- Options: latest versions; exact pins in `pyproject.toml` with a lock file.
- Decision: exact pins plus `uv.lock`; PyTorch from its CPU index; indirect packages released
  within two weeks of locking are capped (`constraint-dependencies`) because uv's `exclude-newer`
  cannot read the PyTorch index; numpy stays at 1.26, so pyarrow is capped below 20. One OpenCV
  build serves both Ultralytics and EasyOCR (`override-dependencies`). Ultralytics runs offline
  with automatic installs off; Hugging Face and Streamlit telemetry are off.
- Risks: a security fix in a capped package needs a deliberate update and a re-run of the tests
  and of `make track` to confirm unchanged results.

### All parameters in configuration files (2026-11-15)
- Context: the lab's own data (stereo camera, other cities) should not need code changes.
- Decision: drive split, detector settings, LiDAR parameters, segment length, no-parking distances
  and OCR settings live in `configs/`; code reads them.
