import datetime as dt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.features import PRECISION_WEIGHT
from biocatalyst.sources import guidance as g

WIN = (dt.date(2026, 11, 1), dt.date(2027, 4, 30))


def test_period_parsing():
    assert g.parse_period("first half of 2027") == (dt.date(2027, 1, 1),
                                                    dt.date(2027, 6, 30))
    assert g.parse_period("Q4 2026") == (dt.date(2026, 10, 1),
                                         dt.date(2026, 12, 31))
    assert g.parse_period("second quarter of 2026") == (dt.date(2026, 4, 1),
                                                        dt.date(2026, 6, 30))
    assert g.parse_period("March 2027") == (dt.date(2027, 3, 1),
                                            dt.date(2027, 3, 31))


def test_vague_period_returns_nothing():
    assert g.parse_period("soon") == (None, None)
    assert g.parse_period("") == (None, None)


def test_guidance_narrows_the_window():
    lo, hi, src = g.narrow(*WIN, dt.date(2027, 1, 1), dt.date(2027, 3, 31))
    assert (lo, hi, src) == (dt.date(2027, 1, 1), dt.date(2027, 3, 31), "guidance")
    assert (hi - lo).days < (WIN[1] - WIN[0]).days


def test_non_overlapping_guidance_is_ignored():
    # A company guiding to a period the trial cannot support is more likely
    # talking about a different programme than reading out early.
    assert g.narrow(*WIN, dt.date(2028, 1, 1), dt.date(2028, 12, 31)) == (
        WIN[0], WIN[1], "trial")


def test_guidance_never_widens():
    lo, hi, _ = g.narrow(*WIN, dt.date(2020, 1, 1), dt.date(2030, 12, 31))
    assert lo >= WIN[0] and hi <= WIN[1]


def test_missing_guidance_keeps_the_window():
    assert g.narrow(*WIN, None, None) == (WIN[0], WIN[1], "trial")


def test_readout_window_is_trusted_less_than_a_stated_quarter():
    # A completion date tested no better than chance at locating the market
    # reaction; a company stating a quarter is telling you more.
    assert PRECISION_WEIGHT["readout_window"] < PRECISION_WEIGHT["quarter"]
    assert PRECISION_WEIGHT["guidance"] > PRECISION_WEIGHT["quarter"]
    assert PRECISION_WEIGHT["guidance"] < PRECISION_WEIGHT["day"]


def test_guidance_regex_catches_common_phrasings():
    for text in ["topline data expected in the first half of 2027",
                 "results are expected in Q4 2026",
                 "readout expected in mid-2027",
                 "data expected by March 2027"]:
        assert g.GUIDANCE_RE.search(text), text


def test_narrow_handles_pandas_nulls():
    # Regression: a date column round-tripped through pandas comes back as
    # NaT, which is not None, so the guard passed and max() raised.
    import numpy as np
    import pandas as pd
    for null in (None, pd.NaT, np.nan, float("nan")):
        assert g.narrow(*WIN, null, null) == (WIN[0], WIN[1], "trial")
    # a real lower bound with a null upper bound still narrows
    lo, hi, src = g.narrow(*WIN, dt.date(2027, 2, 1), pd.NaT)
    assert (lo, src) == (dt.date(2027, 2, 1), "guidance")


def test_narrow_survives_a_null_window():
    import pandas as pd
    assert g.narrow(pd.NaT, pd.NaT, dt.date(2027, 1, 1),
                    dt.date(2027, 3, 31))[2] == "trial"
