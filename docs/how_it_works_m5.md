# How it works: M5, reading parking signs

KITTI is in Germany, so sign reading uses photos of Edmonton parking signs instead. The photos
and their transcriptions are the owner's data; nothing here runs until they exist.

## 1. Photos without metadata

Photos go into `data/signs/` as `sign_001.jpg`, `sign_002.jpg`, ... (a second photo of the same
sign: `sign_001_b.jpg`). `data/` is never committed.

`make strip-exif` writes `data/signs_clean/`: each photo is turned upright using its EXIF
orientation, then saved as a new JPEG with no metadata at all (no GPS, phone or time). The script
checks that the saved file has no EXIF left. YOLO then looks for people in the clean copies and
lists any photo that may show one; licence plates are checked by eye. HEIC photos must be
exported as JPEG first.

## 2. Transcriptions: `labels/signs_groundtruth.csv`

One row per photo, written before looking at any OCR output:

| Column | Format | Example |
| --- | --- | --- |
| file | photo name | `sign_001.jpg` |
| text_lines | the sign's text, line by line, separated by `\|`; unreadable characters `[?]` | `2 HR PARKING \| 9 AM - 6 PM \| MON - SAT` |
| type | `time_limit`, `no_parking`, `no_stopping`, `permit`, `loading_zone`, `pay_parking`, `tow_away` or `unknown` | `time_limit` |
| max_minutes | number, or empty | `120` |
| days | `MON-FRI` for three or more days in a row, `SAT,SUN` for a list or two days, `SUN` for one; empty for every day or not stated | `MON-SAT` |
| start, end | 24-hour `HH:MM`, from the first time range; empty if none | `09:00`, `18:00` |
| exceptions | `HOLIDAYS`, `PERMIT`, or `HOLIDAYS;PERMIT`; empty if none | `HOLIDAYS` |
| permit_zone | zone code, or empty | `12B` |
| arrow | `left`, `right`, `both` or empty | `both` |
| condition | `clear`, `angled`, `snow`, `dirty` or `backlit` | `clear` |

The type is the first that applies in the order above (a sign with both NO STOPPING and NO
PARKING is `no_stopping`; NO PARKING with TOW AWAY is `no_parking`). A time limit with a permit
exception (`2 HR PARKING EXCEPT BY PERMIT ZONE 12`) is `time_limit` with exception `PERMIT` and
zone `12`.

## 3. One detector, three recognisers

`make ocr` (`scripts/ocr_benchmark.py`):

1. **Detect once.** EasyOCR's text detector finds the text boxes on each photo
   (`curbscope/ocr/detect_lines.py`). Boxes are grouped into lines and each line becomes one
   crop: the box around all the words of that line. The detector often leaves out short marks
   such as the dash in `9 AM - 6 PM`; cropping the whole line keeps them.
2. **Recognise the same crops three ways** (`curbscope/ocr/recognizers.py`):
   - **Tesseract** with `--psm 7` (each crop is a single line). It needs the system program:
     `sudo apt install tesseract-ocr`.
   - **EasyOCR**'s own recogniser on the crop.
   - **TrOCR** `microsoft/trocr-base-printed`, a transformer, pinned to one model commit.
     It is accurate on printed text but slow on a CPU (about half a minute per sign).
   Because the crops are identical, differences come from recognition alone.
3. **Score.** Per recogniser: character error rate (CER, with `jiwer`) against the
   transcription, after upper-casing and ignoring line breaks; CPU seconds per photo; and the
   share of photos whose parsed rule equals the transcribed rule exactly. Scores are also split by
   photo condition. The parser is also run on the transcriptions themselves, which shows how much
   of the loss is the parser's and how much the OCR's.

Outputs: `results/ocr_benchmark.json` and `results/ocr_outputs.csv` (every recogniser's text,
CER and parsed rule per photo).

## 4. From text to a rule

`curbscope/ocr/parse_rules.py`, covered by `tests/test_parse_rules.py`:

- **OCR confusions** are fixed only inside runs that contain a digit: O/o to 0, I/l to 1,
  B to 8, S to 5. `I5 MIN` becomes `15 MIN`; `SAT` or `BUS` never change. Permit zones are read
  from the unfixed text, because codes like `12B` are real.
- **Time limits:** `2 HR`, `2 HOUR`, `15 MIN`, `1/2 HR`.
- **Time ranges:** `9AM-6PM`, `8:00 AM TO 5:30 PM`, `NOON-6PM`. A side without AM/PM borrows the
  other side's unless that puts the start after the end (`9-6PM` is 09:00-18:00); with no AM/PM
  at all, a smaller end hour is read as afternoon (`9-5` is 09:00-17:00). Only the first range
  is kept.
- **Days:** ranges (`MON-FRI`, `MONDAY TO FRIDAY`), lists (`SAT & SUN`), `WEEKDAYS`,
  `WEEKENDS`. Days named after `EXCEPT` are removed (`EXCEPT SUNDAYS & HOLIDAYS` gives MON-SAT
  with exception HOLIDAYS).
- **Types:** NO STOPPING, NO PARKING, LOADING ZONE, PERMIT ... ONLY / RESIDENTIAL PARKING,
  PAY PARKING, a time limit, TOW AWAY.
- Arrows are not read; they come from the transcription.

Known gaps: only the first of several time ranges is kept, a letter inside a day name
(`M0N`) is not repaired, and signs with several rules stacked on one plate produce one rule.
