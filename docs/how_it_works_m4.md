# How it works: M4, curb segments

`make segments` turns parked cars into curb numbers per 20 m segment, for each side of the
street (code in `curbscope/segments.py`, stage `segments` in `scripts/run_pipeline.py`).
It needs `configs/frozen_thresholds.yaml` from M3 and stops with a message without it.

## Everything is measured along our path

Our own path (the OXTS positions) is a polyline. Any point in the world is placed on it by:

- **s**: the distance along the path to the nearest point on it (shapely's `project`);
- **side**: the sign of the cross product between the path direction there and the vector to the
  point; positive means left of the direction of travel;
- **offset**: the distance from the path.

A parked vehicle (rule B with the frozen thresholds) counts as being at the curb when its offset
is between 1.5 and 12 m. Its position is the median world position from M3.

## Vehicle length along the curb

All LiDAR points of the vehicle over every frame of its track (the same object points M2 uses)
are projected onto the path direction; the span from the 5th to the 95th percentile is its length.
With fewer than 20 points, or a span outside 2 to 8 m, the length falls back to 4.5 m and the
vehicle is marked `length_source: default`, so the report can say how often this happened.

## Segments

Each side is cut into 20 m segments from the start of the path (the last one may be shorter).
Lengths along the curb are intervals in s:

- **occupied**: the union of the vehicles' intervals `[s - length/2, s + length/2]`. A union,
  because two overlapping cars must not count twice.
- **no parking**: for every detected fire hydrant or stop sign at the curb, 5 m on each side
  along the curb, from `configs/rules_edmonton.yaml` (with sources). Intersections, crosswalks
  and driveways also have rules, but the detector has no class for them, so they are not applied.
- **usable** = segment length - no-parking length.
- **occupancy** = occupied length inside the usable part / usable length. A car standing in a
  no-parking zone is reported separately as `occupied_no_parking_m`, so occupancy never goes
  above 1. A segment with no usable length has no occupancy.
- **vehicles** and **first_seen_s**: the parked vehicles whose centre falls in the segment, and
  the time each was first seen (seconds from the start of the drive). With an ID switch, the
  second track's first-seen time is later than the car's real arrival in view.

## Outputs

- `results/segments_<drive>.csv`: one row per side and segment.
- `results/segments_<drive>.geojson`: segment lines drawn 4 m beside our path on their side
  (on a tight bend the inner offset line can split; the longest part is drawn), with all metrics
  as properties; and one point per vehicle track with a position (`parked` true, false or null).
  Coordinates are longitude, latitude (GeoJSON order) from the M2 conversion. Open it in QGIS.
- `results/segments_summary.json`: per drive, parked vehicles at the curb, how many lengths came
  from LiDAR, no-parking objects, total usable and occupied length, overall occupancy; and the
  SHA-256 of the frozen thresholds used.
