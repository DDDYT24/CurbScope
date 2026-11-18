"""The generated report: numbers come from results files, missing results show as pending."""

import json

import pytest

from curbscope import report
from curbscope.config import load_config

CFG = load_config()
D0, D1, D2 = CFG["drives"]["dev"] + CFG["drives"]["test"]


def write(path, data):
    path.write_text(json.dumps(data))


def geo_result(split):
    stats = {"pairs": 10, "center": {"median": 1.704, "p90": 2.1},
             "footprint": {"median": 0.0, "p90": 0.0}}
    return {"split": split, "vehicle_boxes": 20, "vehicle_boxes_unknown": 3,
            "error_by_range": {"0-10 m": stats, "10-20 m": {"pairs": 0, "center": {
                "median": None, "p90": None}, "footprint": {"median": None, "p90": None}},
                "all": stats}}


def test_report_with_only_geo_results_marks_the_rest_pending(tmp_path):
    write(tmp_path / f"geo_error_{D0}.json", geo_result("dev"))
    text = report.build_report(tmp_path, CFG)
    assert "| 0022 | dev | 10 | 1.70 / 2.10 | 0.00 / 0.00 | 3 / 20 |" in text
    assert "1.70 (10) | - (0)" in text          # an empty range bin prints a dash
    assert report.PENDING["motion"] in text and report.PENDING["ocr"] in text
    assert "Pending: `make track`." in text


def test_motion_segments_and_ocr_tables(tmp_path):
    write(tmp_path / "motion_eval.json", {
        "frozen_sha256": "ab" * 32,
        "frozen": {"rule_a": {"max_shift_over_height": 2.0},
                   "rule_b": {"max_spread_m": 1.5, "max_speed_mps": 1.0}},
        "splits": {"test": {"compared_tracks": 40, "rules": {"rule_b_world": {
            "accuracy": 0.925, "precision_parked": 0.97, "recall_parked": 0.94,
            "confusion": {"parked_as_parked": 32, "moving_as_parked": 1,
                          "parked_as_moving": 2, "moving_as_moving": 5}}}}}})
    write(tmp_path / "segments_summary.json", {"drives": {D2: {
        "path_length_m": 414.2, "parked_at_curb": 44, "length_from_lidar": 13,
        "length_default": 31, "usable_m": 828.5, "occupied_usable_m": 145.6,
        "overall_occupancy": 0.1757, "occupied_no_parking_m": 0.0}}})
    write(tmp_path / "ocr_benchmark.json", {"photos": 48, "parser_on_transcriptions": 0.9,
                                            "recognizers": {"trocr": {
                                                "cer_mean": 0.081, "cer_median": 0.05,
                                                "parse_exact": 0.6,
                                                "cpu_seconds_per_photo": 31.2}}})
    text = report.build_report(tmp_path, CFG)
    assert "| test | rule_b_world | 40 | 92.5% | 97.0% | 94.0% | 32 / 1 / 2 / 5 |" in text
    assert "spread <= 1.5 m" in text
    assert "| 0023 | 414.2 | 44 | 13 / 31 | 828.5 | 145.6 | 17.6% | 0.0 |" in text
    assert "| trocr | 8.1% | 5.0% | 60.0% | 31.20 |" in text and "48 photos" in text


def test_rows_follow_the_config_order(tmp_path):
    write(tmp_path / "tracks_summary.json", {
        "settings": {"weights_sha256": "0" * 64, "ultralytics": "8", "imgsz": 1280,
                     "conf": 0.1},
        "drives": {d: {"frames": 1, "tracks": i, "tracks_by_class": {},
                       "detections_tracked": 0, "detections_raw": 0,
                       "cpu_seconds_per_frame": 0.5} for i, d in enumerate(sorted([D0, D1, D2]))}})
    section = report.tracking_section(tmp_path, CFG)
    order = [line.split("|")[1].strip() for line in section.splitlines()[2:5]]
    assert order == [report._short(D0), report._short(D1), report._short(D2)]


def test_replace_block_is_idempotent_and_needs_markers():
    text = f"intro\n{report.START}\nold\n{report.END}\nrest\n"
    once = report.replace_block(text, "new")
    assert once == f"intro\n{report.START}\nnew\n{report.END}\nrest\n"
    assert report.replace_block(once, "new") == once
    with pytest.raises(ValueError):
        report.replace_block("no markers here", "new")


def test_readme_has_the_markers_and_matches_make_report():
    readme = (report.Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    results = report.Path(__file__).parents[1] / "results"
    assert report.replace_block(readme, report.readme_block(results, CFG)) == readme


def test_mixed_thresholds_and_missing_test_split_are_reported(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    frozen = {"rule_a": {"max_shift_over_height": 2.0},
              "rule_b": {"max_spread_m": 1.5, "max_speed_mps": 1.0}}
    rules = {"rule_b_world": {"accuracy": 1.0, "precision_parked": 1.0, "recall_parked": 1.0,
                              "confusion": {"parked_as_parked": 3, "moving_as_parked": 0,
                                            "parked_as_moving": 0, "moving_as_moving": 1}}}
    write(results / "motion_eval.json", {"frozen_sha256": "a" * 64, "frozen": frozen,
                                         "splits": {"dev": {"compared_tracks": 4,
                                                            "rules": rules}}})
    write(results / "segments_summary.json", {"frozen_sha256": "b" * 64, "drives": {}})
    text = report.motion_section(results, CFG)
    assert "| test | both | - | pending |" in text
    assert "Warning: results come from different thresholds" in text
    write(results / "segments_summary.json", {"frozen_sha256": "a" * 64, "drives": {}})
    assert "Warning" not in report.motion_section(results, CFG)
