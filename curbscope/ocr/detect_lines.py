"""Find text on a sign once, with EasyOCR's detector, and cut it into one crop per text line,
top to bottom.

All three recognisers then read exactly these crops, so differences in the benchmark come from
recognition alone, not from where each tool thinks the text is.
"""

import numpy as np


def make_reader():
    """EasyOCR reader for English on CPU. Downloads its models on first use."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def detect_boxes(reader, image_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Axis-aligned text boxes (x1, y1, x2, y2). Slanted boxes are replaced by their bounds."""
    horizontal, free = reader.detect(image_rgb)
    boxes = [(int(b[0]), int(b[2]), int(b[1]), int(b[3])) for b in horizontal[0]]
    for poly in free[0]:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return [b for b in boxes if b[2] > b[0] and b[3] > b[1]]


def reading_order(boxes) -> list[list[tuple[int, int, int, int]]]:
    """Group boxes into lines (vertical centres closer than half a box height), top to bottom,
    each line left to right."""
    lines: list[list] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        centre, height = (box[1] + box[3]) / 2, box[3] - box[1]
        if lines:
            last = lines[-1]
            last_centre = np.mean([(b[1] + b[3]) / 2 for b in last])
            if abs(centre - last_centre) < 0.5 * max(height, np.mean([b[3] - b[1] for b in last])):
                last.append(box)
                continue
        lines.append([box])
    return [sorted(line, key=lambda b: b[0]) for line in lines]


def line_box(line) -> tuple[int, int, int, int]:
    """One box around all boxes of a line. The detector often leaves out short marks between
    words (the dash in 9 AM - 6 PM); reading the whole line keeps them."""
    return (min(b[0] for b in line), min(b[1] for b in line),
            max(b[2] for b in line), max(b[3] for b in line))


def crop(image: np.ndarray, box, pad: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    return image[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)]


def join_lines(texts: list[str]) -> str:
    """Lines joined by ' | ' (the transcription format); empty lines dropped."""
    return " | ".join(t.strip() for t in texts if t.strip())
