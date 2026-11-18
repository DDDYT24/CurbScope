"""Write results/report.md and the results block of README.md from results/*.json.

Usage: python -m curbscope.report   (or make report)
Every number in the report and in the README's results block comes from a results file written
by a pipeline script. A result that does not exist yet is shown as pending, with what it needs.
"""

import argparse
import hashlib
import json
from pathlib import Path

from curbscope.config import DEFAULT, all_drives, load_config

START, END = "<!-- results:start -->", "<!-- results:end -->"
PENDING = {
    "motion": "Pending: needs the owner's track labels (`make label`) and the frozen thresholds in "
              "`configs/frozen_thresholds.yaml` (`make tune`, then `make evaluate SPLIT=test`).",
    "segments": "Pending: needs `configs/frozen_thresholds.yaml` (see parked or moving), then "
                "`make segments`.",
    "ocr": "Pending: needs the owner's Edmonton sign photos, their transcriptions in "
           "`labels/signs_groundtruth.csv` and the Tesseract program, then `make strip-exif ocr`.",
}


def _load(results: Path, name: str) -> dict | None:
    path = results / name
    return json.loads(path.read_text()) if path.exists() else None


def _m(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _short(drive: str) -> str:
    return drive.rsplit("_", 1)[-1]


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _split(cfg: dict, drive: str) -> str:
    return "test" if drive in cfg["drives"]["test"] else "dev"


def drives_section(results: Path, cfg: dict) -> str:
    rows = []
    for d in all_drives(cfg):
        info = _load(results, f"info_{d}.json")
        if info is None:
            continue
        cars = (info.get("tracklets") or {}).get("by_class", {}).get("Car", "-")
        static = (info.get("tracklets") or {}).get("static_by_class", {}).get("Car", "-")
        rows.append([d, _split(cfg, d), info["frames"], f"{info['duration_s']:.1f}",
                     f"{info['distance_m']:.1f}", f"{cars} ({static})"])
    if not rows:
        return "Pending: `make info DRIVE=<id>` for each drive."
    return _table(["Drive", "Split", "Frames", "Duration (s)", "Distance (m)",
                   "Labelled cars (still)"], rows)


def tracking_section(results: Path, cfg: dict) -> str:
    s = _load(results, "tracks_summary.json")
    if s is None:
        return "Pending: `make track`."
    classes = cfg["detect"]["classes"]
    rows = [[_short(d), v["frames"], v["tracks"],
             *[v["tracks_by_class"].get(c, 0) for c in classes],
             f"{v['detections_tracked']} / {v['detections_raw']}",
             f"{v['cpu_seconds_per_frame']:.2f}"]
            for d in all_drives(cfg) if (v := s["drives"].get(d))]
    st = s["settings"]
    note = (f"YOLO11n COCO weights (SHA-256 `{st['weights_sha256'][:12]}`), Ultralytics "
            f"{st['ultralytics']}, image size {st['imgsz']}, confidence {st['conf']}, ByteTrack "
            "defaults. CPU seconds depend on the machine.")
    return _table(["Drive", "Frames", "Tracks", *classes, "Boxes tracked / detected",
                   "CPU s per frame"], rows) + "\n\n" + note


def geo_section(results: Path, cfg: dict) -> str:
    rows, by_range = [], []
    for d in all_drives(cfg):
        g = _load(results, f"geo_error_{d}.json")
        if g is None:
            continue
        a = g["error_by_range"]["all"]
        rows.append([_short(d), g["split"], a["pairs"],
                     f"{_m(a['center']['median'])} / {_m(a['center']['p90'])}",
                     f"{_m(a['footprint']['median'])} / {_m(a['footprint']['p90'])}",
                     f"{g['vehicle_boxes_unknown']} / {g['vehicle_boxes']}"])
        bins = [k for k in g["error_by_range"] if k != "all"]
        by_range.append([_short(d)] + [
            f"{_m(g['error_by_range'][b]['center']['median'])} "
            f"({g['error_by_range'][b]['pairs']})" for b in bins])
    if not rows:
        return "Pending: `make geo`."
    return (_table(["Drive", "Split", "Matched boxes", "Error to centre, median / p90 (m)",
                    "Error to footprint, median / p90 (m)", "Boxes without position"], rows)
            + "\n\nMedian error to the label centre by range (matched boxes in brackets):\n\n"
            + _table(["Drive"] + bins, by_range)
            + "\n\nLiDAR sees the near side of a car while the label sits at its centre, so the "
              "error to the centre includes up to half a car length by construction; the error "
              "to the footprint is 0 when the point lies on the labelled car.")


def motion_section(results: Path, cfg: dict) -> str:
    e = _load(results, "motion_eval.json")
    if e is None:
        return PENDING["motion"]
    rows = []
    for split, r in e["splits"].items():
        for rule, m in r["rules"].items():
            c = m["confusion"]
            rows.append([split, rule, r["compared_tracks"], _pct(m["accuracy"]),
                         _pct(m["precision_parked"]), _pct(m["recall_parked"]),
                         f"{c['parked_as_parked']} / {c['moving_as_parked']} / "
                         f"{c['parked_as_moving']} / {c['moving_as_moving']}"])
    if "test" not in e["splits"]:
        rows.append(["test", "both", "-", "pending", "-", "-", "`make evaluate SPLIT=test`"])
    f = e["frozen"]
    note = _threshold_warning(results) + (
            f"Frozen thresholds (SHA-256 `{e['frozen_sha256'][:12]}`): rule A box shift / "
            f"height <= {f['rule_a']['max_shift_over_height']}; rule B spread <= "
            f"{f['rule_b']['max_spread_m']} m and speed <= {f['rule_b']['max_speed_mps']} m/s. "
            "Unsure labels and tracks a rule cannot decide are excluded and counted in "
            "results/motion_eval.json.")
    return _table(["Split", "Rule", "Tracks", "Accuracy", "Precision (parked)",
                   "Recall (parked)", "Confusion P→P / M→P / P→M / M→M"], rows) + "\n\n" + note


def _threshold_warning(results: Path) -> str:
    """A warning line when the parked/moving evaluation, the curb segments and the current
    frozen-threshold file do not all carry the same SHA-256 (after a threshold v2, say)."""
    shas = {}
    for name in ("motion_eval.json", "segments_summary.json"):
        data = _load(results, name)
        if data and data.get("frozen_sha256"):
            shas[name] = data["frozen_sha256"]
    frozen = results.parent / "configs" / "frozen_thresholds.yaml"
    if frozen.exists():
        shas["configs/frozen_thresholds.yaml"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    if len(set(shas.values())) > 1:
        parts = ", ".join(f"{k} `{v[:12]}`" for k, v in shas.items())
        return f"**Warning: results come from different thresholds ({parts}).** "
    return ""


def segments_section(results: Path, cfg: dict) -> str:
    s = _load(results, "segments_summary.json")
    if s is None:
        return PENDING["segments"]
    rows = [[_short(d), v["path_length_m"], v["parked_at_curb"],
             f"{v['length_from_lidar']} / {v['length_default']}", v["usable_m"],
             v["occupied_usable_m"], _pct(v["overall_occupancy"]),
             v["occupied_no_parking_m"]]
            for d in all_drives(cfg) if (v := s["drives"].get(d))]
    return _table(["Drive", "Path (m)", "Parked at curb", "Length from LiDAR / default",
                   "Usable curb (m)", "Occupied (m)", "Occupancy", "Occupied in no-parking (m)"],
                  rows) + ("\n\n" + _threshold_warning(results)
                           + "Usable curb counts both sides. Per-segment numbers are in "
                           "results/segments_<drive>.csv and .geojson.")


def ocr_section(results: Path, cfg: dict) -> str:
    o = _load(results, "ocr_benchmark.json")
    if o is None:
        return PENDING["ocr"]
    rows = [[name, _pct(r["cer_mean"]), _pct(r["cer_median"]), _pct(r["parse_exact"]),
             f"{r['cpu_seconds_per_photo']:.2f}"] for name, r in o["recognizers"].items()]
    return (_table(["Recogniser", "CER mean", "CER median", "Rule exactly right",
                    "CPU s per photo"], rows)
            + f"\n\n{o['photos']} photos. The parser alone, on the transcriptions, gets "
              f"{_pct(o['parser_on_transcriptions'])} of the rules exactly right.")


SECTIONS = [
    ("Drives", drives_section),
    ("Detection and tracking", tracking_section),
    ("Position against KITTI's 3D labels", geo_section),
    ("Parked or moving", motion_section),
    ("Curb segments", segments_section),
    ("Edmonton parking sign OCR", ocr_section),
]


def build_report(results: Path, cfg: dict) -> str:
    parts = ["# CurbScope report", "",
             "Generated by `make report` from the files in `results/`. Nothing here is typed by "
             "hand."]
    for title, section in SECTIONS:
        parts += ["", f"## {title}", "", section(results, cfg)]
    return "\n".join(parts) + "\n"


def readme_block(results: Path, cfg: dict) -> str:
    """The README's results: the same tables, one heading level down."""
    parts = []
    for title, section in SECTIONS[2:]:
        parts += [f"### {title}", "", section(results, cfg), ""]
    parts.append("Full tables: [results/report.md](results/report.md).")
    return "\n".join(parts)


def replace_block(text: str, block: str) -> str:
    if START not in text or END not in text:
        raise ValueError(f"README.md needs the markers {START} and {END}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = Path(args.results_dir)
    (results / "report.md").write_text(build_report(results, cfg), encoding="utf-8")
    readme = Path(args.readme)
    readme.write_text(replace_block(readme.read_text(encoding="utf-8"),
                                    readme_block(results, cfg)), encoding="utf-8")
    print(f"wrote {results / 'report.md'} and the results block of {readme}")


if __name__ == "__main__":
    main()
