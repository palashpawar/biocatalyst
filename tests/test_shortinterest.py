import datetime as dt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.backtest import pit
from biocatalyst.sources import shortinterest as si


def test_publication_lags_settlement():
    # FINRA discloses ~8 business days after settlement. Treating settlement
    # as the knowledge date would hand a study a position nobody could see.
    settle = dt.date(2026, 8, 31)
    pub = si.publication_date(settle)
    assert pub > settle
    assert (pub - settle).days >= 12      # 10 business days spans two weekends


def test_days_to_cover_drives_the_tiers():
    low = si.squeeze_metrics(1e6, 1.0, 1e8)
    mod = si.squeeze_metrics(1e6, 3.0, 1e8)
    elev = si.squeeze_metrics(1e6, 6.0, 1e8)
    high = si.squeeze_metrics(1e6, 12.0, 1e8)
    assert [low["squeeze_risk"], mod["squeeze_risk"],
            elev["squeeze_risk"], high["squeeze_risk"]] == [0, 1, 2, 3]
    assert high["squeeze_label"] == "high"


def test_heavy_short_percentage_escalates_regardless_of_dtc():
    # 35% of shares out with a thin days-to-cover is still a crowded exit.
    m = si.squeeze_metrics(35e6, 1.0, 100e6)
    assert m["squeeze_risk"] == 3


def test_short_build_is_relative_to_prior_reading():
    m = si.squeeze_metrics(255272, 1.0, 14e6, prior_short=61079)
    assert m["short_build"] > 3.0


def test_missing_inputs_do_not_crash():
    m = si.squeeze_metrics(None, None, None, None)
    assert m["squeeze_risk"] == 0
    assert m["short_pct_outstanding"] is None
    assert m["short_build"] is None


def test_percent_is_of_outstanding_not_float():
    # Float is not available free; the field is named for what it measures.
    m = si.squeeze_metrics(20e6, 2.0, 100e6)
    assert m["short_pct_outstanding"] == 0.2
    assert "float" not in " ".join(m.keys())


ROWS = [
    {"settlement_date": "2026-07-31", "published_date": "2026-08-14",
     "shares_short": 100.0, "days_to_cover": 2.0, "shares_short_prior": 90.0},
    {"settlement_date": "2026-08-31", "published_date": "2026-09-14",
     "shares_short": 200.0, "days_to_cover": 5.0, "shares_short_prior": 100.0},
]


def test_pit_uses_publication_not_settlement():
    # On 1 Sep the August settlement exists but is not public for two weeks.
    r = pit.short_interest_at(ROWS, dt.date(2026, 9, 1))
    assert r["settlement_date"] == "2026-07-31"

    r = pit.short_interest_at(ROWS, dt.date(2026, 9, 20))
    assert r["settlement_date"] == "2026-08-31"


def test_pit_before_any_publication():
    assert pit.short_interest_at(ROWS, dt.date(2026, 1, 1)) is None


def test_pit_on_the_publication_day_itself():
    r = pit.short_interest_at(ROWS, dt.date(2026, 8, 14))
    assert r is not None and r["settlement_date"] == "2026-07-31"
