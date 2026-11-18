"""Parked or moving: per-track features, the two rules, and how they score against labels.

Rule A (image baseline): how far the box centre moves in the image, relative to the box height.
From a moving car a parked car also moves in the image, so this rule is expected to struggle.
Rule B (world): how much the track's world position spreads around its median, plus the speed
implied by its first and last thirds. A parked car stays put in world coordinates.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from curbscope.detect_track import track_class

LABELS = ("parked", "moving", "unsure")
LABEL_COLUMNS = ["track_id", "label", "note"]
FROZEN_KEYS = {"rule_a": ["max_shift_over_height"], "rule_b": ["max_spread_m", "max_speed_mps"]}


def track_features(tracks: pd.DataFrame, located: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per track with the scores both rules use.

    tracks: M1 CSV (boxes). located: M2 CSV (world positions, status known/unknown).
    cfg keys: min_frames_rule_a, min_known_rule_b.
    """
    classes = track_class(tracks)
    known = located[located["status"] == "known"]
    rows = []
    for track_id, boxes in tracks.groupby("track_id"):
        boxes = boxes.sort_values("frame")
        row = {
            "track_id": int(track_id),
            "class": classes[track_id],
            "frames": int(len(boxes)),
            "first_frame": int(boxes["frame"].iloc[0]),
            "last_frame": int(boxes["frame"].iloc[-1]),
            "first_seen_s": float(boxes["timestamp"].iloc[0]),
            "box_shift_over_height": np.nan,
            "known_positions": 0,
            "spread_m": np.nan,
            "speed_mps": np.nan,
            "median_x": np.nan,
            "median_y": np.nan,
        }
        if len(boxes) >= cfg["min_frames_rule_a"]:
            cx = ((boxes["x1"] + boxes["x2"]) / 2).to_numpy()
            cy = ((boxes["y1"] + boxes["y2"]) / 2).to_numpy()
            shift = np.hypot(cx - cx[0], cy - cy[0]).max()
            row["box_shift_over_height"] = float(shift / (boxes["y2"] - boxes["y1"]).median())

        pos = known[known["track_id"] == track_id].sort_values("frame")
        row["known_positions"] = int(len(pos))
        if len(pos) >= cfg["min_known_rule_b"]:
            xy = pos[["x", "y"]].to_numpy()
            centre = np.median(xy, axis=0)
            row["median_x"], row["median_y"] = float(centre[0]), float(centre[1])
            row["spread_m"] = float(np.quantile(np.linalg.norm(xy - centre, axis=1), 0.9))
            third = max(len(pos) // 3, 1)
            head, tail = pos.iloc[:third], pos.iloc[-third:]
            dt = tail["timestamp"].median() - head["timestamp"].median()
            if dt > 0:
                moved = np.linalg.norm(np.median(tail[["x", "y"]].to_numpy(), axis=0)
                                       - np.median(head[["x", "y"]].to_numpy(), axis=0))
                row["speed_mps"] = float(moved / dt)
        rows.append(row)
    return pd.DataFrame(rows)


def rule_a(features: pd.DataFrame, max_shift_over_height: float) -> pd.Series:
    """True = parked, False = moving, NA = cannot decide (track too short)."""
    score = features["box_shift_over_height"]
    return (score <= max_shift_over_height).astype("boolean").mask(score.isna())


def rule_b(features: pd.DataFrame, max_spread_m: float, max_speed_mps: float) -> pd.Series:
    """True = parked, False = moving, NA = cannot decide (too few world positions).

    A track whose speed cannot be computed (all positions at one timestamp) is decided by
    spread alone.
    """
    spread, speed = features["spread_m"], features["speed_mps"]
    slow = (speed <= max_speed_mps) | speed.isna()
    return ((spread <= max_spread_m) & slow).astype("boolean").mask(spread.isna())


def load_labels(path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"note": str}, keep_default_na=False)
    bad = sorted(set(labels["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"{path}: unknown labels {bad}; allowed: {LABELS}")
    if labels["track_id"].duplicated().any():
        raise ValueError(f"{path}: a track is labelled twice")
    return labels


def load_split(drives: list[str], results_dir, labels_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and labels of several drives, each with a drive column (track ids repeat
    across drives). Raises if a drive has no features or no label file yet."""
    features, labels, missing = [], [], []
    for drive in drives:
        f_path = Path(results_dir) / f"motion_features_{drive}.csv"
        l_path = Path(labels_dir) / f"track_labels_{drive}.csv"
        if not f_path.exists():
            raise FileNotFoundError(f"{f_path} not found; run `make motion` first")
        if not l_path.exists():
            missing.append(str(l_path))
            continue
        features.append(pd.read_csv(f_path).assign(drive=drive))
        labels.append(load_labels(l_path).assign(drive=drive))
    if missing:
        raise FileNotFoundError(f"no labels yet: {missing}; label them with `make label`")
    return pd.concat(features, ignore_index=True), pd.concat(labels, ignore_index=True)


def load_frozen(path) -> tuple[dict, str]:
    """The owner's frozen thresholds and the file's SHA-256."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Choose thresholds on the dev drives with `make tune`, write "
            "them to this file and commit it before evaluating."
        )
    data = path.read_bytes()
    frozen = yaml.safe_load(data) or {}
    for rule, keys in FROZEN_KEYS.items():
        for key in keys:
            value = (frozen.get(rule) or {}).get(key)
            if not isinstance(value, int | float):
                raise ValueError(f"{path}: {rule}.{key} must be a number")
    return frozen, hashlib.sha256(data).hexdigest()


def parked_by_track(features: pd.DataFrame, frozen: dict) -> dict[int, bool | None]:
    """Rule B's decision per track as plain Python values: True, False, or None (undecided).
    A pandas boolean Series yields numpy booleans, which `is True` never matches."""
    decided = rule_b(features, frozen["rule_b"]["max_spread_m"],
                     frozen["rule_b"]["max_speed_mps"])
    return {int(t): (None if pd.isna(v) else bool(v))
            for t, v in zip(features["track_id"], decided, strict=True)}


def predict(features: pd.DataFrame, frozen: dict) -> dict[str, pd.Series]:
    return {
        "rule_a_image": rule_a(features, frozen["rule_a"]["max_shift_over_height"]),
        "rule_b_world": rule_b(features, frozen["rule_b"]["max_spread_m"],
                               frozen["rule_b"]["max_speed_mps"]),
    }


def score(labels: pd.DataFrame, predictions: dict[str, pd.Series], features: pd.DataFrame
          ) -> dict:
    """Compare rules on the same tracks: labelled parked or moving, and decided by every rule.

    labels and features both have drive and track_id columns (ids repeat across drives).
    predictions: rule name -> Series aligned with the rows of features.
    """
    keys = ["drive", "track_id"]
    table = features[keys].copy()
    for name, pred in predictions.items():
        table[name] = pred.to_numpy()
    table = table.merge(labels[keys + ["label"]], on=keys, how="inner")
    clear = table[table["label"].isin(["parked", "moving"])]
    decided = clear.dropna(subset=list(predictions))
    out = {
        "labelled_tracks": int(len(table)),
        "unsure": int((table["label"] == "unsure").sum()),
        "not_decided_by_every_rule": int(len(clear) - len(decided)),
        "compared_tracks": int(len(decided)),
        "label_counts": {k: int(v) for k, v in decided["label"].value_counts().items()},
        "rules": {},
    }
    truth = (decided["label"] == "parked").to_numpy()
    for name in predictions:
        pred = decided[name].astype(bool).to_numpy()
        tp, fp = int(np.sum(pred & truth)), int(np.sum(pred & ~truth))
        fn, tn = int(np.sum(~pred & truth)), int(np.sum(~pred & ~truth))
        n = tp + fp + fn + tn
        out["rules"][name] = {
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "precision_parked": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall_parked": round(tp / (tp + fn), 4) if tp + fn else None,
            "confusion": {"parked_as_parked": tp, "moving_as_parked": fp,
                          "parked_as_moving": fn, "moving_as_moving": tn},
        }
    return out
