"""Curb segments: usable length, occupied length, occupancy, vehicle count, first-seen times.

Everything is measured along our own path. Our path (OXTS positions) is a polyline; a parked
vehicle is placed on it by its distance along the path (s) and its side (left or right). Each
side is cut into fixed-length segments. Lengths along the curb are intervals [start, end] in s.
"""

import json

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from curbscope import geo

# --- intervals along the path -------------------------------------------------------------


def union(intervals) -> list[tuple[float, float]]:
    """Merge overlapping or touching intervals."""
    merged = []
    for lo, hi in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def clip(intervals, lo: float, hi: float) -> list[tuple[float, float]]:
    return [(max(a, lo), min(b, hi)) for a, b in union(intervals) if min(b, hi) > max(a, lo)]


def subtract(intervals, cuts) -> list[tuple[float, float]]:
    """Parts of intervals not covered by cuts."""
    out = []
    cuts = union(cuts)
    for a, b in union(intervals):
        start = a
        for c, d in cuts:
            if d <= start or c >= b:
                continue
            if c > start:
                out.append((start, c))
            start = max(start, d)
        if start < b:
            out.append((start, b))
    return out


def length(intervals) -> float:
    return float(sum(b - a for a, b in union(intervals)))


# --- placing things on the path -----------------------------------------------------------


def path_line(path_xy: np.ndarray) -> LineString:
    return LineString(np.asarray(path_xy)[:, :2])


def place(line: LineString, xy) -> tuple[float, float, np.ndarray]:
    """Distance along the path, signed offset (positive = left of travel) and the unit
    tangent at that point."""
    p = Point(float(xy[0]), float(xy[1]))
    s = line.project(p)
    a = np.array(line.interpolate(max(s - 0.5, 0)).coords[0])
    b = np.array(line.interpolate(min(s + 0.5, line.length)).coords[0])
    tangent = (b - a) / np.linalg.norm(b - a)
    foot = np.array(line.interpolate(s).coords[0])
    d = np.array([p.x, p.y]) - foot
    cross = tangent[0] * d[1] - tangent[1] * d[0]
    return float(s), float(np.sign(cross) * np.linalg.norm(d)), tangent


def vehicle_length(points_xy: np.ndarray, tangent: np.ndarray, cfg: dict) -> tuple[float, str]:
    """Length along the curb from LiDAR points gathered over the track's frames: the 5th to
    95th percentile span along the path. Falls back to the configured default when there are
    too few points or the span is implausible, and says so."""
    lo, hi = cfg["vehicle_length_range_m"]
    if len(points_xy) >= cfg["min_length_points"]:
        along = np.asarray(points_xy) @ tangent
        span = float(np.quantile(along, 0.95) - np.quantile(along, 0.05))
        if lo <= span <= hi:
            return span, "lidar"
    return float(cfg["default_vehicle_length_m"]), "default"


# --- segments -----------------------------------------------------------------------------


def build_segments(path_length: float, segment_m: float, vehicles: pd.DataFrame,
                   buffers: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every segment on both sides.

    vehicles: parked vehicles with s, side ('left'/'right'), length_m, track_id, first_seen_s.
    buffers: no-parking zones with s, side, half_width_m (zone is s +/- half_width_m).
    """
    edges = np.arange(0.0, path_length, segment_m)
    rows = []
    for side in ("left", "right"):
        cars = vehicles[vehicles["side"] == side]
        car_iv = [(r.s - r.length_m / 2, r.s + r.length_m / 2) for r in cars.itertuples()]
        zones = buffers[buffers["side"] == side]
        zone_iv = [(r.s - r.half_width_m, r.s + r.half_width_m) for r in zones.itertuples()]
        for start in edges:
            end = min(start + segment_m, path_length)
            no_parking = clip(zone_iv, start, end)
            usable = subtract([(start, end)], no_parking)
            occupied = clip(car_iv, start, end)
            occupied_usable = length(subtract(occupied, no_parking))
            inside = cars[(cars["s"] >= start) & (cars["s"] < end)].sort_values("first_seen_s")
            usable_m = length(usable)
            rows.append({
                "side": side,
                "start_m": round(float(start), 3),
                "end_m": round(float(end), 3),
                "segment_m": round(float(end - start), 3),
                "no_parking_m": round(length(no_parking), 3),
                "usable_m": round(usable_m, 3),
                "occupied_m": round(length(occupied), 3),
                "occupied_usable_m": round(occupied_usable, 3),
                "occupied_no_parking_m": round(length(occupied) - occupied_usable, 3),
                "occupancy": round(occupied_usable / usable_m, 4) if usable_m > 0 else None,
                "vehicles": int(len(inside)),
                "track_ids": ";".join(str(int(t)) for t in inside["track_id"]),
                "first_seen_s": ";".join(f"{t:.1f}" for t in inside["first_seen_s"]),
            })
    return pd.DataFrame(rows)


# --- GeoJSON ------------------------------------------------------------------------------


def _plain(value):
    """JSON-safe value: NaN and pandas NA become null, numpy scalars become Python ones."""
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _lonlat(xy: np.ndarray, origin) -> list[list[float]]:
    latlon = geo.world_to_latlon(np.asarray(xy, dtype=float), *origin)
    return [[round(lon, 7), round(lat, 7)] for lat, lon in latlon]


def geojson(line: LineString, segs: pd.DataFrame, vehicles: pd.DataFrame, origin,
            display_offset_m: float) -> dict:
    """Segment lines drawn beside our path (left or right) and vehicle points, in lon/lat."""
    features = []
    for r in segs.to_dict("records"):
        piece = substring(line, r["start_m"], r["end_m"])
        if piece.length == 0:
            continue
        shifted = piece.offset_curve(display_offset_m if r["side"] == "left"
                                     else -display_offset_m)
        if shifted.geom_type == "MultiLineString":
            # On a tight bend the offset line on the inside folds into several parts;
            # draw the longest one.
            shifted = max(shifted.geoms, key=lambda g: g.length)
        if shifted.is_empty:
            continue
        props = {k: _plain(v) for k, v in r.items()}
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "LineString",
                                      "coordinates": _lonlat(np.array(shifted.coords), origin)}})
    for r in vehicles.to_dict("records"):
        props = {k: _plain(r[k]) for k in ("track_id", "class", "parked", "first_seen_s", "side",
                                           "length_m", "length_source") if k in r}
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Point",
                                      "coordinates": _lonlat([[r["x"], r["y"]]], origin)[0]}})
    return {"type": "FeatureCollection", "features": features}


def write_geojson(data: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
