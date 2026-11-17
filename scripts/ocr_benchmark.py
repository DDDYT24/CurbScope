"""Compare Tesseract, EasyOCR and TrOCR on the same text crops of the owner's sign photos.

Usage: python scripts/ocr_benchmark.py
Needs data/signs_clean/ (from scripts/strip_exif.py) and labels/signs_groundtruth.csv (the
owner's transcriptions). Writes results/ocr_benchmark.json and results/ocr_outputs.csv.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
from PIL import Image

from curbscope.config import DEFAULT, load_config
from curbscope.ocr import detect_lines, parse_rules, recognizers

GROUNDTRUTH_COLUMNS = ["file", "text_lines", "type", "max_minutes", "days", "start", "end",
                       "exceptions", "permit_zone", "arrow", "condition"]


def comparable(text: str) -> str:
    """Upper case, line separators as spaces, single spaces: CER compares characters, not
    where a recogniser or the transcription broke the lines."""
    return " ".join(str(text).upper().replace("|", " ").split())


def load_groundtruth(path: Path, signs_dir: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found. Transcribe every photo first (see docs/how_it_works_m5.md).")
    truth = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_cols = [c for c in GROUNDTRUTH_COLUMNS if c not in truth.columns]
    if missing_cols:
        sys.exit(f"{path} lacks columns {missing_cols}")
    missing = [f for f in truth["file"] if not (signs_dir / f).exists()]
    if missing:
        sys.exit(f"photos listed in {path} but not in {signs_dir}/: {missing[:5]}; "
                 "run scripts/strip_exif.py")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--signs-dir")
    parser.add_argument("--groundtruth")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    cfg = load_config(args.config)["ocr"]
    signs_dir = Path(args.signs_dir or cfg["signs_dir"])
    truth = load_groundtruth(Path(args.groundtruth or cfg["groundtruth"]), signs_dir)

    reader = detect_lines.make_reader()
    readers = {
        "tesseract": recognizers.tesseract(cfg["tesseract_config"]),
        "easyocr": recognizers.easyocr_recognizer(reader),
        "trocr": recognizers.trocr(cfg["trocr_model"], cfg["trocr_revision"]),
    }

    rows = []
    for sign in truth.to_dict("records"):
        image = np.asarray(Image.open(signs_dir / sign["file"]).convert("RGB"))
        lines = detect_lines.reading_order(detect_lines.detect_boxes(reader, image))
        crops = [detect_lines.crop(image, detect_lines.line_box(line), cfg["crop_pad_px"])
                 for line in lines]
        reference = comparable(sign["text_lines"])
        for name, read in readers.items():
            start = time.process_time()
            text = detect_lines.join_lines([read(c) for c in crops])
            seconds = time.process_time() - start
            parsed = parse_rules.parse(text)
            hypothesis = comparable(text)
            cer = jiwer.cer(reference, hypothesis) if reference else None
            rows.append({"file": sign["file"], "condition": sign["condition"],
                         "recognizer": name, "text_lines_detected": len(lines),
                         "text": text, "cer": cer, "cpu_seconds": round(seconds, 3),
                         "parse_ok": parse_rules.matches(parsed, sign),
                         "parsed": json.dumps({k: parsed[k] for k in parse_rules.FIELDS})})
    outputs = pd.DataFrame(rows)

    transcription_ok = [parse_rules.matches(parse_rules.parse(r["text_lines"]), r)
                        for r in truth.to_dict("records")]
    result = {
        "photos": int(len(truth)),
        "conditions": {k: int(v) for k, v in truth["condition"].value_counts().items()},
        "parser_on_transcriptions": round(float(np.mean(transcription_ok)), 4),
        "recognizers": {},
    }
    for name, part in outputs.groupby("recognizer"):
        result["recognizers"][name] = {
            "cer_mean": round(float(part["cer"].mean()), 4),
            "cer_median": round(float(part["cer"].median()), 4),
            "cpu_seconds_per_photo": round(float(part["cpu_seconds"].mean()), 3),
            "parse_exact": round(float(part["parse_ok"].mean()), 4),
            "cer_by_condition": {c: round(float(g["cer"].mean()), 4)
                                 for c, g in part.groupby("condition")},
        }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(out / "ocr_outputs.csv", index=False)
    (out / "ocr_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
