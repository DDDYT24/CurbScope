# How it works: M6, report and dashboard

## The report

`make report` (`curbscope/report.py`) reads the JSON files in `results/` and writes
`results/report.md`: the drives, detection and tracking, position error against KITTI's labels
(overall and by range), parked or moving, curb segments and sign OCR. It also rewrites the block
between `<!-- results:start -->` and `<!-- results:end -->` in `README.md` with the last four of
those sections. Nothing in either is typed by hand: a section whose results do not exist yet says
what it is waiting for (labels, frozen thresholds, photos). A test checks that the README block
equals what `make report` would write, so a stale README fails the tests.

## The dashboard

`make dashboard` opens a Streamlit app (`apps/dashboard.py`) with four tabs:

- **Overview** (per drive): distance, tracks, tracks with a position, parked vehicles at the curb;
  an OpenStreetMap map with the curb segments coloured by occupancy (pale to full blue, grey where
  nothing can be parked) and vehicle points (filled = parked); occupancy per segment for each side.
  Before the thresholds are frozen, the map shows every located track. The overview needs only
  `results/`, not the KITTI images.
- **Tracks** (per drive): any frame with its tracked boxes and IDs, coloured parked, moving or
  undecided once the thresholds are frozen. Needs the KITTI images.
- **Signs**: a photo, the text each recogniser read, its CER, whether the parsed rule was right,
  and the parsed rule.
- **Evaluation**: the report's tables and figures, built by the same functions as the report, so
  the two can never disagree.

The theme (`.streamlit/config.toml`) keeps colour on data and the one action, and turns off
Streamlit's usage statistics.

## `make all`

Runs track, geo and motion; then evaluate and segments only if `configs/frozen_thresholds.yaml`
exists; then strip-exif and OCR only if `labels/signs_groundtruth.csv` exists; then the report.
Skipped steps print why.
