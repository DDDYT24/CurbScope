"""EXIF stripping, reading order and benchmark input checks (no OCR models needed)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from curbscope.ocr import detect_lines

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strip_removes_exif_and_applies_orientation(tmp_path):
    src, dst = tmp_path / "sign_001.jpg", tmp_path / "clean.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6          # orientation: rotate 90 degrees to view upright
    exif[0x8825] = {2: (53.0, 31.0, 0.0), 1: "N"}  # GPS block with a latitude
    Image.new("RGB", (40, 20), "white").save(src, exif=exif.tobytes())
    assert len(Image.open(src).getexif())

    load_script("strip_exif").strip(src, dst)
    clean = Image.open(dst)
    assert len(clean.getexif()) == 0 and "exif" not in clean.info
    assert clean.size == (20, 40)   # turned upright
    assert not dst.with_suffix(".tmp").exists()


def test_reading_order_groups_lines_top_to_bottom_left_to_right():
    boxes = [(120, 52, 200, 80), (10, 10, 100, 40), (10, 50, 100, 82), (110, 12, 180, 38)]
    assert detect_lines.reading_order(boxes) == [
        [(10, 10, 100, 40), (110, 12, 180, 38)],
        [(10, 50, 100, 82), (120, 52, 200, 80)],
    ]
    assert detect_lines.join_lines(["2 HR", " ", "MON-SAT "]) == "2 HR | MON-SAT"


def test_line_box_spans_the_gap_between_words():
    # "9 AM" and "6 PM" detected apart; the line box keeps the dash between them
    assert detect_lines.line_box([(10, 12, 60, 40), (90, 10, 140, 42)]) == (10, 10, 140, 42)


def test_crop_is_clipped_to_the_image():
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    assert detect_lines.crop(image, (0, 0, 10, 10), pad=4).shape == (14, 14, 3)


def test_comparable_ignores_case_and_line_breaks():
    comparable = load_script("ocr_benchmark").comparable
    assert comparable("2 hr parking |  9 AM-6 PM") == comparable("2 HR PARKING 9 AM-6 PM")


def test_benchmark_refuses_without_transcriptions(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ocr_benchmark.py"),
                        "--groundtruth", str(tmp_path / "none.csv"), "--signs-dir",
                        str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "Transcribe every photo" in r.stderr


def test_benchmark_refuses_photos_not_stripped_yet(tmp_path):
    gt = tmp_path / "gt.csv"
    gt.write_text("file,text_lines,type,max_minutes,days,start,end,exceptions,permit_zone,"
                  "arrow,condition\nsign_001.jpg,NO PARKING,no_parking,,,,,,,,clear\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ocr_benchmark.py"),
                        "--groundtruth", str(gt), "--signs-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "strip_exif" in r.stderr
