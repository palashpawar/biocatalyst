import datetime as dt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.sources.bpc import parse_catalyst_date as p


def test_us_slash_date():
    assert p("09/21/2026") == (dt.date(2026, 9, 21), dt.date(2026, 9, 21), "day")


def test_iso_date_survives_hyphen_normalisation():
    # Regression: hyphens were stripped before the ISO regex ran.
    assert p("2025-03-17") == (dt.date(2025, 3, 17), dt.date(2025, 3, 17), "day")


def test_iso_year_month():
    lo, hi, prec = p("2026-11")
    assert (lo, hi, prec) == (dt.date(2026, 11, 1), dt.date(2026, 11, 30), "month")


def test_month_part():
    lo, hi, prec = p("Mid Sep 2026")
    assert prec == "month_part" and lo.day == 11 and hi.day == 20


def test_half_both_orderings():
    assert p("2H 2026")[0] == dt.date(2026, 7, 1)
    assert p("H2 2026")[0] == dt.date(2026, 7, 1)
    assert p("1H 2026")[1] == dt.date(2026, 6, 30)


def test_quarter():
    lo, hi, prec = p("Q4 2026")
    assert (lo, hi, prec) == (dt.date(2026, 10, 1), dt.date(2026, 12, 31), "quarter")


def test_bare_year():
    assert p("2026") == (dt.date(2026, 1, 1), dt.date(2026, 12, 31), "year")


def test_unparseable():
    assert p("TBD")[2] == "unknown"
    assert p("")[2] == "unknown"
    assert p(None)[2] == "unknown"


def test_invalid_calendar_date():
    assert p("02/31/2026")[2] == "unknown"


def test_february_month_end_not_overflowed():
    lo, hi, prec = p("Late Feb 2026")
    assert hi == dt.date(2026, 2, 28) and prec == "month_part"


def test_month_name_with_day_keeps_the_day():
    # Regression: PDUFA dates in 8-Ks read "November 15, 2026". These fell
    # through to the month-only branch, silently became the 1st of the month,
    # and were then labelled day-precision by the caller. Near-term ones were
    # rounded backwards past today and dropped as already-passed.
    assert p("November 15, 2026") == (dt.date(2026, 11, 15),
                                      dt.date(2026, 11, 15), "day")
    assert p("April 1 2027")[2] == "day"
    assert p("Sep 30, 2026") == (dt.date(2026, 9, 30),
                                 dt.date(2026, 9, 30), "day")


def test_day_before_month_ordering():
    assert p("15 April 2027") == (dt.date(2027, 4, 15),
                                  dt.date(2027, 4, 15), "day")


def test_ordinal_suffixes():
    assert p("March 3rd, 2027") == (dt.date(2027, 3, 3),
                                    dt.date(2027, 3, 3), "day")


def test_month_only_still_spans_the_month():
    # The day-bearing branch must not swallow bare month/year strings.
    assert p("September 2026") == (dt.date(2026, 9, 1),
                                   dt.date(2026, 9, 30), "month")


def test_impossible_day_rejected():
    assert p("February 30 2026")[2] == "unknown"
