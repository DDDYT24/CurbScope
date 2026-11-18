"""Helpers for the track labelling page (apps/label_tracks.py), kept here so they are tested."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from curbscope.motion import LABEL_COLUMNS, LABELS


def label_path(labels_dir, drive: str) -> Path:
    return Path(labels_dir) / f"track_labels_{drive}.csv"


def read_labels(path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(path, dtype={"note": str}, keep_default_na=False)


def save_label(path, track_id: int, label: str, note: str = "") -> pd.DataFrame:
    """Add or replace one track's label. Written to a temporary file first, then moved into
    place, so a crash never leaves a half-written label file."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    labels = read_labels(path)
    labels = labels[labels["track_id"].astype(int) != int(track_id)]
    row = pd.DataFrame([{"track_id": int(track_id), "label": label, "note": note.strip()}])
    labels = pd.concat([labels, row], ignore_index=True).sort_values("track_id")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    labels.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return labels


def pick_frames(frames: np.ndarray, n: int) -> list[int]:
    """Up to n frames spread evenly over a track, always including the first and last."""
    frames = np.sort(np.asarray(frames))
    if len(frames) <= n:
        return frames.tolist()
    return frames[np.linspace(0, len(frames) - 1, n).round().astype(int)].tolist()


def crop_box(image: np.ndarray, box, pad: float = 0.3) -> np.ndarray:
    """The box with some context around it, clipped to the image."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    h, w = image.shape[:2]
    left, top = max(int(x1 - dx), 0), max(int(y1 - dy), 0)
    right, bottom = min(int(x2 + dx) + 1, w), min(int(y2 + dy) + 1, h)
    return image[top:bottom, left:right]
