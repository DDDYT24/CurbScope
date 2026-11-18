"""Parking sign parser. Cases cover the sign formats listed in the project brief; tests built from
the owner's transcriptions are added once labels/signs_groundtruth.csv exists."""

import pytest

from curbscope.ocr import parse_rules as pr

RULES = [
    # text, type, max_minutes, days, start, end, exceptions, permit_zone
    ("2 HR PARKING | 9 AM - 6 PM | MON - SAT", "time_limit", 120, "MON-SAT", "09:00", "18:00",
     None, None),
    ("2 HOUR PARKING 8:00 AM TO 5:30 PM MONDAY TO FRIDAY", "time_limit", 120, "MON-FRI",
     "08:00", "17:30", None, None),
    ("15 MIN PARKING", "time_limit", 15, None, None, None, None, None),
    ("1/2 HR PARKING 9AM-6PM", "time_limit", 30, None, "09:00", "18:00", None, None),
    ("NO PARKING ANYTIME", "no_parking", None, None, None, None, None, None),
    ("NO PARKING 7AM-9AM MON-FRI", "no_parking", None, "MON-FRI", "07:00", "09:00", None, None),
    ("NO STOPPING 3:30PM-6PM MON-FRI", "no_stopping", None, "MON-FRI", "15:30", "18:00", None,
     None),
    ("NO STOPPING NO PARKING", "no_stopping", None, None, None, None, None, None),
    ("NO PARKING TOW AWAY ZONE", "no_parking", None, None, None, None, None, None),
    ("TOW AWAY ZONE", "tow_away", None, None, None, None, None, None),
    ("LOADING ZONE 15 MIN", "loading_zone", 15, None, None, None, None, None),
    ("LOADING ONLY 7AM-11AM", "loading_zone", None, None, "07:00", "11:00", None, None),
    ("RESIDENTIAL PARKING ONLY | PERMIT ZONE 7", "permit", None, None, None, None, None, "7"),
    ("PERMIT HOLDERS ONLY ZONE 12B", "permit", None, None, None, None, None, "12B"),
    ("2 HR PARKING EXCEPT BY PERMIT ZONE 12", "time_limit", 120, None, None, None, "PERMIT",
     "12"),
    ("2 HR PARKING PERMIT HOLDERS EXCEPTED", "time_limit", 120, None, None, None, "PERMIT", None),
    ("PAY PARKING 2 HR MAX 9AM-6PM", "pay_parking", 120, None, "09:00", "18:00", None, None),
    ("PAY AT METER", "pay_parking", None, None, None, None, None, None),
    ("2 HR PARKING 9AM-6PM EXCEPT SUNDAYS & HOLIDAYS", "time_limit", 120, "MON-SAT", "09:00",
     "18:00", "HOLIDAYS", None),
    ("1 HR PARKING MON-SAT EXCEPT HOLIDAYS", "time_limit", 60, "MON-SAT", None, None,
     "HOLIDAYS", None),
    ("NO PARKING SAT & SUN", "no_parking", None, "SAT,SUN", None, None, None, None),
    ("NO PARKING WEEKDAYS 8-5", "no_parking", None, "MON-FRI", "08:00", "17:00", None, None),
    ("NO PARKING WEEKENDS", "no_parking", None, "SAT,SUN", None, None, None, None),
    ("NO PARKING MON WED FRI", "no_parking", None, "MON,WED,FRI", None, None, None, None),
    ("2 HR 11-2PM", "time_limit", 120, None, "11:00", "14:00", None, None),
    ("NO PARKING NOON-6PM", "no_parking", None, None, "12:00", "18:00", None, None),
    ("NO PARKING 10PM-6AM", "no_parking", None, None, "22:00", "06:00", None, None),
    ("NO PARKING 8-18", "no_parking", None, None, "08:00", "18:00", None, None),
    ("NO PARKING 7AM-9AM & 4PM-6PM", "no_parking", None, None, "07:00", "09:00", None, None),
    ("2 hr parking 9 a.m. - 6 p.m. mon - sat", "time_limit", 120, "MON-SAT", "09:00", "18:00",
     None, None),
    ("NO PARKING DAILY 9AM-5PM", "no_parking", None, None, "09:00", "17:00", None, None),
    ("NO PARKING MONTHLY STREET CLEANING", "no_parking", None, None, None, None, None, None),
    ("WELCOME TO GARNEAU", "unknown", None, None, None, None, None, None),
]


@pytest.mark.parametrize("text,type_,minutes,days,start,end,exceptions,zone", RULES)
def test_parse(text, type_, minutes, days, start, end, exceptions, zone):
    got = pr.parse(text)
    assert (got["type"], got["max_minutes"], got["days"], got["start"], got["end"],
            got["exceptions"], got["permit_zone"]) == (type_, minutes, days, start, end,
                                                       exceptions, zone)
    assert got["raw_text"] == text


@pytest.mark.parametrize("ocr,fixed", [
    ("I5 MIN", "15 MIN"),           # I read as 1 next to a digit
    ("2 HR 9AM-6PM", "2 HR 9AM-6PM"),
    ("1O-6", "10-6"),               # O read as 0 inside a number
    ("8:3O AM", "8:30 AM"),
    ("S:30 PM", "5:30 PM"),         # S read as 5 inside a time
    ("B AM", "B AM"),               # a lone letter is not a number
    ("BUS LOT SAT", "BUS LOT SAT"),  # words without digits never change
    ("l2 HR", "12 HR"),
])
def test_fix_confusions_only_inside_numbers(ocr, fixed):
    assert pr.fix_confusions(ocr) == fixed


def test_confusions_are_fixed_before_parsing():
    got = pr.parse("I5 MIN PARKING 8:3OAM-S:3OPM MON-FRI")
    assert (got["max_minutes"], got["start"], got["end"], got["days"]) == (15, "08:30", "17:30",
                                                                           "MON-FRI")


def test_digit_inside_a_day_name_is_not_repaired():
    # M0N has a zero but is not a number; the parser does not guess it is MON.
    assert pr.parse("NO PARKING M0N-FRI")["days"] == "FRI"


def test_matches_normalises_the_transcription():
    parsed = pr.parse("2 HR PARKING | 9 AM - 6 PM | MON - SAT EXCEPT HOLIDAYS")
    row = {"type": "Time_Limit", "max_minutes": "120", "days": "mon-sat", "start": "9:00",
           "end": "18:00", "exceptions": "holidays", "permit_zone": ""}
    assert pr.matches(parsed, row)
    assert not pr.matches(parsed, {**row, "max_minutes": "60"})
    assert not pr.matches(parsed, {**row, "exceptions": ""})


def test_normalise_truth_treats_blank_and_nan_as_none():
    truth = pr.normalise_truth({"type": "no_parking", "max_minutes": "nan", "days": " ",
                                "start": None, "end": "", "exceptions": "PERMIT;holidays",
                                "permit_zone": "7a"})
    assert truth == {"type": "no_parking", "max_minutes": None, "days": None, "start": None,
                     "end": None, "exceptions": "HOLIDAYS;PERMIT", "permit_zone": "7A"}
