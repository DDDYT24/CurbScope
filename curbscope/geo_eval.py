"""Compare located vehicles with KITTI's 3D tracklet labels.

Per frame, labelled vehicles and located detections are paired by horizontal world distance
(Hungarian algorithm, pairs further apart than a fixed gate are not matches). Two errors are
reported per pair:
- center: horizontal distance to the label's box centre (the main number);
- footprint: horizontal distance to the label's box outline on the ground, 0 inside.
LiDAR only sees the near side of a car, so center error has a built-in bias of up to half a
car length; footprint error removes that bias and shows whether the point is on the car.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from curbscope import geo


def tracklets_by_frame(tracklets, T_w_velo: np.ndarray, types: list[str]) -> pd.DataFrame:
    """One row per labelled vehicle per frame, in world coordinates.

    T_w_velo is (N, 4, 4), one pose per frame. Tracklet poses are box bottom centres in the
    Velodyne frame; rz is the heading around the Velodyne z axis.
    """
    rows = []
    for index, t in enumerate(tracklets):
        if t.object_type not in types:
            continue
        for frame, pose in zip(t.frames, t.poses, strict=True):
            if frame >= len(T_w_velo):
                continue
            T = T_w_velo[frame]
            x, y, _ = geo.apply_transform(T, pose[:3])[0]
            yaw = pose[5] + np.arctan2(T[1, 0], T[0, 0])
            rows.append({
                "frame": int(frame), "tracklet": index, "type": t.object_type,
                "x": x, "y": y, "yaw": yaw, "l": t.l, "w": t.w,
                "range_m": float(np.hypot(pose[0], pose[1])),
            })
    return pd.DataFrame(rows, columns=["frame", "tracklet", "type", "x", "y", "yaw", "l", "w",
                                       "range_m"])


def match(det_xy: np.ndarray, gt_xy: np.ndarray, max_m: float) -> list[tuple[int, int, float]]:
    """Hungarian matching on horizontal distance; returns (det index, gt index, distance)."""
    if len(det_xy) == 0 or len(gt_xy) == 0:
        return []
    dist = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    # Pairs beyond the gate get a cost no real pair can reach, then are dropped.
    cost = np.where(dist <= max_m, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(dist[r, c])) for r, c in zip(rows, cols, strict=True)
            if dist[r, c] <= max_m]


def footprint_distance(p_xy, center_xy, yaw: float, length: float, width: float) -> float:
    """Distance from a point to a rotated ground rectangle (0 when inside)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = np.asarray(p_xy, float) - np.asarray(center_xy, float)
    along, across = c * dx + s * dy, -s * dx + c * dy  # into the box's own axes
    out_along = max(abs(along) - length / 2, 0.0)
    out_across = max(abs(across) - width / 2, 0.0)
    return float(np.hypot(out_along, out_across))


def match_drive(located: pd.DataFrame, labels: pd.DataFrame, max_m: float) -> pd.DataFrame:
    """Pairs for a whole drive. located needs frame, x, y; labels from tracklets_by_frame."""
    pairs = []
    for frame, gts in labels.groupby("frame"):
        dets = located[located["frame"] == frame]
        found = match(dets[["x", "y"]].to_numpy(), gts[["x", "y"]].to_numpy(), max_m)
        for i, j, d in found:
            det, gt = dets.iloc[i], gts.iloc[j]
            pairs.append({
                "frame": int(frame), "track_id": int(det["track_id"]),
                "tracklet": int(gt["tracklet"]), "range_m": gt["range_m"],
                "center_error_m": d,
                "footprint_error_m": footprint_distance(
                    (det["x"], det["y"]), (gt["x"], gt["y"]), gt["yaw"], gt["l"], gt["w"]),
            })
    return pd.DataFrame(pairs, columns=["frame", "track_id", "tracklet", "range_m",
                                        "center_error_m", "footprint_error_m"])


def _stats(values: pd.Series) -> dict:
    if values.empty:
        return {"median": None, "p90": None}
    return {"median": round(float(values.median()), 3),
            "p90": round(float(values.quantile(0.9)), 3)}


def error_by_range(pairs: pd.DataFrame, edges: list[float]) -> dict:
    """Median and 90th percentile of both errors per range bin, plus everything beyond."""
    bins = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    bins.append((edges[-1], np.inf))
    out = {}
    for lo, hi in bins:
        name = f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f"{lo:g}+ m"
        part = pairs[(pairs["range_m"] >= lo) & (pairs["range_m"] < hi)]
        out[name] = {"pairs": int(len(part)),
                     "center": _stats(part["center_error_m"]),
                     "footprint": _stats(part["footprint_error_m"])}
    out["all"] = {"pairs": int(len(pairs)),
                  "center": _stats(pairs["center_error_m"]),
                  "footprint": _stats(pairs["footprint_error_m"])}
    return out


def box_corners_velo(pose: np.ndarray, h: float, w: float, length: float) -> np.ndarray:
    """8 corners (8, 3) of a tracklet box in the Velodyne frame, for drawing."""
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * length / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * h
    c, s = np.cos(pose[5]), np.sin(pose[5])
    return np.column_stack([c * x - s * y, s * x + c * y, z]) + pose[:3]
