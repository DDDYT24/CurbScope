"""Copy the parking sign photos without EXIF metadata (GPS position, phone, time).

Usage: python scripts/strip_exif.py   (reads data/signs/, writes data/signs_clean/)
Each photo is first turned upright using its EXIF orientation, then saved as a new JPEG that
carries no metadata. The originals are not touched. YOLO then looks for people in the clean
copies and lists any photo that may show one; licence plates are checked by eye.
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

from curbscope.config import DEFAULT, load_config

SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


def strip(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        upright = ImageOps.exif_transpose(img).convert("RGB")
    # Write to a temporary name first, so a crash never leaves a half-written photo.
    tmp = dst.with_suffix(".tmp")
    upright.save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dst)
    with Image.open(dst) as check:
        if len(check.getexif()) or "exif" in check.info:
            raise RuntimeError(f"{dst} still has EXIF data")


def people(paths: list[Path], cfg: dict) -> list[tuple[str, float]]:
    from ultralytics import YOLO

    model = YOLO(cfg["detect"]["weights"])
    person = [i for i, n in model.names.items() if n == "person"]
    found = []
    for path in paths:
        result = model.predict(str(path), classes=person, conf=cfg["ocr"]["person_conf"],
                               verbose=False)[0]
        if len(result.boxes):
            found.append((path.name, float(result.boxes.conf.max())))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT))
    args = parser.parse_args()
    cfg = load_config(args.config)
    src_dir, dst_dir = Path(cfg["ocr"]["raw_dir"]), Path(cfg["ocr"]["signs_dir"])
    photos = sorted(p for p in src_dir.glob("*") if p.suffix.lower() in SUFFIXES)
    if not photos:
        sys.exit(f"no photos in {src_dir}/ yet; name them sign_001.jpg, sign_002.jpg, ...")
    heic = [p.name for p in photos if p.suffix.lower() == ".heic"]
    if heic:
        sys.exit(f"HEIC photos cannot be read here; export them as JPEG first: {heic}")

    stems = [p.stem for p in photos]
    clash = sorted({s for s in stems if stems.count(s) > 1})
    if clash:
        sys.exit(f"several photos share a name and would overwrite each other: {clash}; "
                 "rename them first")

    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for src in photos:
        dst = dst_dir / f"{src.stem}.jpg"
        strip(src, dst)
        out.append(dst)
    print(f"{len(out)} photos written to {dst_dir}/ without EXIF")

    flagged = people(out, cfg)
    if flagged:
        print("Possible people (check, crop or retake before publishing anything):")
        for name, conf in flagged:
            print(f"  {name}  confidence {conf:.2f}")
    else:
        print("YOLO found no people. Check licence plates by eye.")


if __name__ == "__main__":
    main()
