"""Detect vehicles and curb objects with YOLO and give them stable IDs with ByteTrack.

The YOLO part is isolated in make_yolo_tracker(); everything else works on plain arrays,
so the tests can feed fake detections without loading a model.
"""

import hashlib
import io
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

COLUMNS = ["frame", "timestamp", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"]
GIF_MAX_BYTES = 5 * 1024 * 1024


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_weights(path: str | Path, expected_sha256: str) -> str:
    """Refuse to run with missing or unexpected weights. Returns the hash."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run `make weights`")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {actual}, config expects {expected_sha256}")
    return actual


def class_ids(names: dict[int, str], wanted: list[str]) -> list[int]:
    """COCO class names -> the model's class ids."""
    by_name = {name: i for i, name in names.items()}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise ValueError(f"classes not in the model: {missing}")
    return [by_name[w] for w in wanted]


def boxes_to_rows(frame: int, timestamp: float, boxes, names: dict[int, str]) -> list[dict]:
    """Rows for one frame. boxes has numpy xyxy (N, 4), conf, cls and id (None if no tracks)."""
    if boxes.id is None:
        return []
    return [
        {
            "frame": frame,
            "timestamp": timestamp,
            "track_id": int(tid),
            "class": names[int(cls)],
            "confidence": float(conf),
            "x1": float(xyxy[0]),
            "y1": float(xyxy[1]),
            "x2": float(xyxy[2]),
            "y2": float(xyxy[3]),
        }
        for xyxy, conf, cls, tid in zip(boxes.xyxy, boxes.conf, boxes.cls, boxes.id, strict=True)
    ]


def track_frames(
    frames: Iterable[tuple[int, float, np.ndarray]],
    track_fn: Callable[[np.ndarray], object],
    names: dict[int, str],
    on_frame: Callable[[int, np.ndarray, list[dict]], None] | None = None,
) -> pd.DataFrame:
    """Run track_fn on (frame, timestamp, RGB image) in order and collect the tracked boxes."""
    rows = []
    for frame, timestamp, image_rgb in frames:
        # Ultralytics treats numpy images as BGR, like OpenCV.
        frame_rows = boxes_to_rows(frame, timestamp, track_fn(image_rgb[..., ::-1]), names)
        rows.extend(frame_rows)
        if on_frame is not None:
            on_frame(frame, image_rgb, frame_rows)
    return pd.DataFrame(rows, columns=COLUMNS)


def track_class(tracks: pd.DataFrame) -> pd.Series:
    """One class per track: the class it was detected as most often (ties: alphabetical)."""
    counts = tracks.groupby(["track_id", "class"]).size().rename("n").reset_index()
    counts = counts.sort_values(["track_id", "n", "class"], ascending=[True, False, True])
    return counts.drop_duplicates("track_id").set_index("track_id")["class"]


def summarize(tracks: pd.DataFrame, n_frames: int, raw_detections: int, seconds: float) -> dict:
    per_track = track_class(tracks)
    return {
        "frames": n_frames,
        "cpu_seconds_per_frame": round(seconds / max(n_frames, 1), 3),
        "detections_raw": raw_detections,
        "detections_tracked": int(len(tracks)),
        "detections_without_track": raw_detections - int(len(tracks)),
        "tracks": int(per_track.size),
        "tracks_by_class": {k: int(v) for k, v in per_track.value_counts().sort_index().items()},
        "detections_by_class": {
            k: int(v) for k, v in tracks["class"].value_counts().sort_index().items()
        },
    }


def make_yolo_tracker(cfg_detect: dict):
    """Return (track_fn, names, counter). counter["raw"] counts detections before tracking."""
    from ultralytics import YOLO

    model = YOLO(cfg_detect["weights"])
    ids = class_ids(model.names, cfg_detect["classes"])
    counter = {"raw": 0}

    def count_raw(predictor) -> None:
        # Registered before the tracker's own callback, so it sees the detections
        # before ByteTrack drops the ones it does not match.
        counter["raw"] += sum(len(r.boxes) for r in predictor.results)

    model.add_callback("on_predict_postprocess_end", count_raw)

    def track_fn(image_bgr: np.ndarray):
        result = model.track(
            image_bgr,
            persist=True,
            tracker=cfg_detect["tracker"],
            classes=ids,
            conf=cfg_detect["conf"],
            imgsz=cfg_detect["imgsz"],
            verbose=False,
        )[0]
        return result.boxes.numpy()

    return track_fn, model.names, counter


def color_for(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def draw_boxes(image_rgb: np.ndarray, rows: list[dict]) -> np.ndarray:
    out = np.ascontiguousarray(image_rgb).copy()
    for r in rows:
        p1, p2 = (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"]))
        color = color_for(r["track_id"])
        # Drawn at full resolution, then the GIF is shrunk to about half: keep lines and text thick.
        cv2.rectangle(out, p1, p2, color, 3)
        cv2.putText(out, str(r["track_id"]), (p1[0], max(p1[1] - 6, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


def gif_bytes(frames_rgb: list[np.ndarray], width: int, fps: float) -> bytes:
    images = []
    for f in frames_rgb:
        height = round(f.shape[0] * width / f.shape[1])
        images.append(Image.fromarray(cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)))
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=round(1000 / fps), loop=0, optimize=True)
    return buf.getvalue()


def write_gif(frames_rgb: list[np.ndarray], path: str | Path, width: int, fps: float) -> int:
    data = gif_bytes(frames_rgb, width, fps)
    if len(data) > GIF_MAX_BYTES:
        raise ValueError(f"GIF is {len(data)} bytes, over 5 MB; lower width or use fewer frames")
    Path(path).write_bytes(data)
    return len(data)
