"""Turn parking sign text into a machine-readable rule.

Output: {type, max_minutes, days, start, end, exceptions, permit_zone, raw_text}
- type: no_stopping, no_parking, loading_zone, permit, pay_parking, time_limit, tow_away, unknown
  (the first that applies, in that order).
- max_minutes: the time limit in minutes, or None.
- days: "MON-FRI" for a run of days, "SAT,SUN" for a list, "SUN" for one day; None when the
  sign restricts no days (every day, DAILY, ANYTIME, or not stated).
- start, end: "HH:MM" in 24-hour time, from the first time range on the sign; None if none.
- exceptions: "HOLIDAYS", "PERMIT" or both joined by ";"; None if none.
- permit_zone: the zone after ZONE or PERMIT NO. on a sign that mentions a permit; None if none.
Arrows are not read from text; they come from the owner's labels.
"""

import re

DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
TYPE_PATTERNS = [
    ("no_stopping", r"\bNO\s+STOPPING\b"),
    ("no_parking", r"\bNO\s+PARKING\b"),
    ("loading_zone", r"\bLOADING\s+(?:ZONE|ONLY)\b"),
    ("permit", r"\bPERMIT\s+(?:HOLDERS\s+|PARKING\s+)?ONLY\b|\bRESIDENT(?:IAL|S)?\s+PARKING\b"),
    ("pay_parking", r"\bPAY\s+(?:PARKING|HERE|AT\s+(?:METER|STATION))\b|\bPAY\s+STATION\b"),
]
TOW_AWAY = r"\bTOW[\s-]*AWAY\b"
FIELDS = ["type", "max_minutes", "days", "start", "end", "exceptions", "permit_zone"]


def fix_confusions(text: str) -> str:
    """Inside runs that contain a digit (numbers and times), read O/o as 0, I/l as 1, B as 8
    and S as 5. Words without digits are left alone, so SAT, BUS or LOT never change."""
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "B": "8", "S": "5"})

    def repair(match: re.Match) -> str:
        run = match.group(0)
        return run.translate(table) if any(c.isdigit() for c in run) else run

    return re.sub(r"[0-9OoIlBS:.]+", repair, text)


def normalise(text: str, fix: bool = True) -> str:
    """Upper case, plain dashes, AM/PM without dots, single spaces; optionally with
    fix_confusions first. Permit zones are read without it, since zones like 12B are real."""
    text = (fix_confusions(text) if fix else text).upper()
    text = text.replace("|", " ").replace("–", "-").replace("—", "-")
    text = re.sub(r"\b([AP])\.\s*M\.?", r"\1M", text)
    return re.sub(r"\s+", " ", text).strip()


def _max_minutes(text: str) -> int | None:
    m = re.search(r"\b(?:(\d{1,2})\s+)?1/2\s*(?:HR|HOUR)\b", text)
    if m:
        return int(m.group(1) or 0) * 60 + 30
    m = re.search(r"\b(\d{1,3})\s*(?:HR|HRS|HOUR|HOURS|H)\b", text)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"\b(\d{1,3})\s*(?:MIN|MINS|MINUTE|MINUTES)\b", text)
    return int(m.group(1)) if m else None


_TIME = r"(?:NOON|MIDNIGHT|\d{1,2}(?:[:.]\d{2})?\s*(?:AM|PM)?)"
_RANGE = re.compile(rf"(?<![\d/])({_TIME})\s*(?:-|TO)\s*({_TIME})(?![\d/])")


def _split_time(token: str) -> tuple[int, int, str | None]:
    token = token.replace(" ", "")
    if token == "NOON":
        return 12, 0, "PM"
    if token == "MIDNIGHT":
        return 12, 0, "AM"
    m = re.fullmatch(r"(\d{1,2})(?:[:.](\d{2}))?(AM|PM)?", token)
    return int(m.group(1)), int(m.group(2) or 0), m.group(3)


def _minutes(hour: int, minute: int, ampm: str | None) -> int:
    if ampm == "AM":
        hour = 0 if hour == 12 else hour
    elif ampm == "PM":
        hour = hour if hour == 12 else hour + 12
    return hour * 60 + minute


def _time_range(text: str) -> tuple[str | None, str | None]:
    """First time range on the sign. A side without AM/PM borrows the other side's, unless that
    would put the start after the end (9-6PM is 09:00-18:00). With no AM/PM at all, hours
    above 12 are 24-hour time and a smaller end hour is read as afternoon (9-5 is 09:00-17:00)."""
    for m in _RANGE.finditer(text):
        (h1, m1, p1), (h2, m2, p2) = _split_time(m.group(1)), _split_time(m.group(2))
        if h1 > 24 or h2 > 24 or m1 > 59 or m2 > 59:
            continue
        if p1 is None and p2 is not None:
            p1 = p2 if _minutes(h1, m1, p2) < _minutes(h2, m2, p2) else "AM"
        elif p2 is None and p1 is not None:
            p2 = p1 if _minutes(h2, m2, p1) > _minutes(h1, m1, p1) else "PM"
        start, end = _minutes(h1, m1, p1), _minutes(h2, m2, p2)
        if p1 is None and p2 is None and h2 <= 12 and end <= start:
            end += 12 * 60
        return f"{start // 60:02d}:{start % 60:02d}", f"{end // 60:02d}:{end % 60:02d}"
    return None, None


# A day name or abbreviation, never the start of a longer word (MONTHLY is not Monday).
_DAY = (r"\b(MON|TUE|WED|THU|FRI|SAT|SUN)"
        r"(?:DAYS?|NESDAYS?|SDAYS?|RSDAYS?|URDAYS?|S|R|RS)?\.?(?![A-Z])")


def _days_in(text: str) -> set[int]:
    days: set[int] = set()
    if re.search(r"\bWEEKDAYS?\b", text):
        days |= set(range(5))
    if re.search(r"\bWEEKENDS?\b", text):
        days |= {5, 6}
    for m in re.finditer(rf"{_DAY}\s*(?:-|TO|THROUGH|THRU)\s*{_DAY}", text):
        a, b = DAYS.index(m.group(1)), DAYS.index(m.group(2))
        days |= {(a + i) % 7 for i in range((b - a) % 7 + 1)}
    text = re.sub(rf"{_DAY}\s*(?:-|TO|THROUGH|THRU)\s*{_DAY}", " ", text)
    days |= {DAYS.index(m.group(1)) for m in re.finditer(_DAY, text)}
    return days


def _format_days(days: set[int]) -> str | None:
    if not days or len(days) == 7:
        return None
    ordered = sorted(days)
    if len(ordered) > 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{DAYS[ordered[0]]}-{DAYS[ordered[-1]]}"
    return ",".join(DAYS[d] for d in ordered)


def _permit_zone(text: str) -> str | None:
    if "PERMIT" not in text:
        return None
    m = re.search(r"\bZONE\s*#?\s*([A-Z]?\d{1,3}[A-Z]?)\b(?!\s*(?:MIN|HR|HOUR|H\b))", text)
    if m:
        return m.group(1)
    m = re.search(r"\bPERMIT\s*(?:NO\.?|#)\s*([A-Z]?\d{1,3}[A-Z]?)\b", text)
    return m.group(1) if m else None


def _split_except(text: str) -> tuple[str, str]:
    """(rule text, exception clause). The clause starts at EXCEPT (not EXCEPTED) and ends where
    the hours or a day range begin, since signs often put the exception on its own line before
    them: in "NO PARKING EXCEPT BY PERMIT 8AM-5PM MON-FRI" the days MON-FRI belong to the rule."""
    parts = re.split(r"\bEXCEPT\b", text, maxsplit=1)
    if len(parts) == 1:
        return text, ""
    before, after = parts
    end = re.search(rf"{_RANGE.pattern}|{_DAY}\s*(?:-|TO|THROUGH|THRU)\s*{_DAY}", after)
    if end is None:
        return before, after
    return before + " " + after[end.start():], after[:end.start()]


def parse(raw_text: str) -> dict:
    text = normalise(raw_text)
    main, except_part = _split_except(text)

    sign_type = next((t for t, p in TYPE_PATTERNS if re.search(p, text)), None)
    max_minutes = _max_minutes(text)
    if sign_type is None:
        sign_type = "time_limit" if max_minutes else (
            "tow_away" if re.search(TOW_AWAY, text) else "unknown")

    days = _days_in(main)
    excluded = _days_in(except_part)
    if excluded:
        days = (days or set(range(7))) - excluded

    exceptions = []
    if re.search(r"\bHOLIDAYS?\b", except_part):
        exceptions.append("HOLIDAYS")
    if re.search(r"\bPERMIT", except_part) or re.search(
            r"\bPERMIT\s+HOLDERS\s+(?:EXCEPTED|EXEMPT)\b", text):
        exceptions.append("PERMIT")

    start, end = _time_range(text)
    return {
        "type": sign_type,
        "max_minutes": max_minutes,
        "days": _format_days(days),
        "start": start,
        "end": end,
        "exceptions": ";".join(sorted(exceptions)) or None,
        "permit_zone": _permit_zone(normalise(raw_text, fix=False)),
        "raw_text": raw_text,
    }


def normalise_truth(row: dict) -> dict:
    """A ground-truth row (strings from the CSV, empty meaning none) in parse()'s format."""
    def clean(value):
        value = "" if value is None else str(value).strip()
        return None if value in ("", "nan", "None") else value

    out = {f: clean(row.get(f)) for f in FIELDS}
    if out["type"]:
        out["type"] = out["type"].lower()
    if out["max_minutes"]:
        out["max_minutes"] = int(float(out["max_minutes"]))
    if out["days"]:
        out["days"] = out["days"].upper().replace(" ", "")
    for key in ("start", "end"):
        if out[key]:
            hour, minute = out[key].split(":")
            out[key] = f"{int(hour):02d}:{int(minute):02d}"
    if out["exceptions"]:
        out["exceptions"] = ";".join(sorted(p.strip().upper()
                                            for p in out["exceptions"].split(";")))
    if out["permit_zone"]:
        out["permit_zone"] = out["permit_zone"].upper()
    return out


def matches(parsed: dict, truth_row: dict) -> bool:
    """True when every rule field equals the owner's transcription."""
    truth = normalise_truth(truth_row)
    return all(parsed[f] == truth[f] for f in FIELDS)
