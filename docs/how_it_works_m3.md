# How it works: M3, parked or moving

## Two rules, compared on hand labels

For every track, `make motion` computes the numbers both rules use
(`results/motion_features_<drive>.csv`, code in `curbscope/motion.py`):

- **Rule A, image baseline.** The largest distance the box centre moves from where it started,
  divided by the median box height. A small value means the box barely moved in the image.
  Needs at least 3 frames.
- **Rule B, world.** The 90th percentile of the distance between each world position (from M2)
  and the track's median position (the *spread*), plus the *speed* implied by the median
  positions of the track's first and last thirds. Needs at least 3 world positions.

A track a rule cannot decide (too short, too few LiDAR positions) is counted, not guessed.

From a moving car a parked car slides across the image, often by many box heights, while its
world position barely changes. Rule A is there to show how much that matters.

The world spread of a parked car is not zero: LiDAR sees its back, then its side, then its front
as we drive past, and the median of those surface points moves by up to about a metre.

## Labels come first

`make label` opens a page (`apps/label_tracks.py`) that shows, for one track at a time, crops of
its box over time, one full frame, and its world trajectory next to our own path. Buttons record
*parked*, *moving* or *unsure*, with an optional note (for example an ID switch), into
`labels/track_labels_<drive>.csv`. The page never shows what the rules decided. The labelling
rules are written in `docs/labeling_guide.md`.

## Thresholds are chosen on dev drives and frozen

- `make tune` scores both rules over a grid of thresholds on the **dev drives only**
  (`results/tune_dev.json`, `results/tune_dev.png`). It refuses the test drive.
- The chosen thresholds, the selection rule and the date go into
  `configs/frozen_thresholds.yaml`, which is committed:

  ```yaml
  rule_a: {max_shift_over_height: <number>}
  rule_b: {max_spread_m: <number>, max_speed_mps: <number>}
  selection: <how they were chosen>
  date: <YYYY-MM-DD>
  ```

- `make evaluate SPLIT=test` refuses to run unless that file exists, is committed and has no
  uncommitted change, so the thresholds provably came before the test result. The file's SHA-256
  is stored in `results/motion_eval.json`.

## Scoring

Both rules are scored on the same tracks: labelled parked or moving, and decided by both rules.
Unsure labels and undecided tracks are excluded and counted. For each rule: accuracy, precision
and recall for *parked*, and the confusion matrix. The test drive has very few moving cars, so
accuracy alone says little; read it with the confusion matrix.
