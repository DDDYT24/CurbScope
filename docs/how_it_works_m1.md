# How it works: M1, detecting and tracking vehicles

## Detection: YOLO

YOLO looks at one image and returns boxes, each with a class and a confidence between 0 and 1.
CurbScope uses the smallest model, YOLO11n, with weights trained on the COCO dataset, and keeps
six COCO classes: car, motorcycle, bus, truck, fire hydrant and stop sign. Hydrants and stop
signs are kept because Edmonton forbids parking within 5 m of them (used in M4).

The model resizes each image before running. KITTI images are 1242 x 375 pixels; at the
default size of 640 a car 40 m away is only a few pixels wide, so `imgsz` is 1280.

## Tracking: ByteTrack

Detection alone has no memory: frame 10's car and frame 11's car are unrelated boxes.
ByteTrack links them so the same car keeps the same `track_id`:

1. A Kalman filter predicts where each existing track's box should be in the new frame,
   assuming roughly constant velocity in the image.
2. High-confidence boxes (at least 0.25) are matched to the predicted boxes by overlap (IoU).
3. Tracks still unmatched get a second chance with low-confidence boxes (0.1 to 0.25). This is
   ByteTrack's main idea: a car half hidden behind a tree often scores low, and throwing those
   boxes away would break the track.
4. A high-confidence box that matches nothing starts a new track. A track with no match is kept
   alive for 30 frames (3 s) in case the car reappears, then dropped.

That is why `conf` is 0.1: the tracker needs to see the low-score boxes. Every setting is in
`configs/bytetrack.yaml`, copied from Ultralytics' defaults and not tuned on any drive, so
running M1 on the test drive does not leak information.

Two consequences to remember:

- The boxes in the CSV are the tracker's boxes: the Kalman filter's estimate after the update,
  slightly smoother than the raw detection.
- Detections that no track takes are dropped by Ultralytics before we see them. A callback that
  runs before the tracker counts all detections, so the summary reports how many were dropped
  (`detections_without_track`).

A track's class can change from frame to frame (a van may be called car, then truck).
`track_class` gives each track the class it was detected as most often.

## Outputs

- `results/tracks_<drive>.csv`: one row per tracked box: `frame, timestamp, track_id, class,
  confidence, x1, y1, x2, y2` (pixels in the rectified left colour image; timestamp in seconds
  since the drive's first frame).
- `results/tracks_summary.json`: per drive, the number of tracks per class, detections before
  and after tracking, and CPU seconds per frame; plus the Ultralytics version and the weights'
  SHA-256 so the run can be repeated exactly.
- `results/demo_2011_09_26_drive_0022.gif`: frames 0 to 80 of drive 0022, every second frame,
  each track in its own colour with its ID.

## Reproducibility and privacy

`make weights` downloads the weights from the Ultralytics release page and checks the SHA-256
in the config; a different file stops the run. `make track` then runs with `YOLO_OFFLINE=1`
(no network, so no usage analytics) and `YOLO_AUTOINSTALL=False` (no packages installed behind
your back). Inference on CPU gives the same boxes on every run.
