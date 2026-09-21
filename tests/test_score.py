import sys, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.score import classify

BASE = {
    "runway_at_catalyst": 24.0, "runway_months": 24.0, "ret_20d": 0.0,
    "move_ratio": 1.0, "implied_move": 0.3, "expected_move": 0.3, "loa": 0.4,
    "stage": "Phase 3", "ta": "other", "designations": "", "microcap": False,
    "optionable": True, "is_binary": True, "date_confidence": 1.0,
    "illiquid": False, "materiality": 1.5, "baseline_move": 0.2,
}


def row(**kw):
    return pd.Series({**BASE, **kw})


def test_out_of_cash_microcap_is_dilution_short():
    r = classify(row(runway_at_catalyst=-2.0, microcap=True))
    assert r["setup"] == "DILUTION_SHORT" and r["lean"] == "short"


def test_rich_straddle_is_vol_sell():
    r = classify(row(move_ratio=2.0, implied_move=0.6))
    assert r["setup"] == "VOL_SELL_RICH"


def test_cheap_straddle_is_vol_buy():
    r = classify(row(move_ratio=0.5, implied_move=0.15))
    assert r["setup"] == "VOL_BUY_CHEAP"


def test_vague_date_lowers_conviction():
    sharp = classify(row(runway_at_catalyst=-2.0, microcap=True, date_confidence=1.0))
    vague = classify(row(runway_at_catalyst=-2.0, microcap=True, date_confidence=0.15))
    assert vague["conviction"] < sharp["conviction"]


def test_illiquidity_discounts_conviction():
    liquid = classify(row(runway_at_catalyst=-2.0, microcap=True))
    thin = classify(row(runway_at_catalyst=-2.0, microcap=True, illiquid=True))
    assert thin["conviction"] < liquid["conviction"]
    assert "thin" in thin["reasons"]


def test_missing_data_does_not_crash():
    r = classify(row(runway_at_catalyst=float("nan"), ret_20d=float("nan"),
                     move_ratio=float("nan"), loa=float("nan")))
    assert r["setup"] in ("NO_EDGE", "BASE_RATE_LONG")


def test_conviction_within_bounds():
    for mr in (0.1, 1.0, 5.0):
        assert 0 <= classify(row(move_ratio=mr))["conviction"] <= 100


def test_immaterial_catalyst_never_becomes_a_vol_trade():
    # Regression: a Phase 3 readout worth ~1% of a mega-cap was being compared
    # to its straddle and flagged as rich/cheap volatility. It is neither --
    # the straddle is pricing ordinary equity noise, not the event.
    rich = classify(row(move_ratio=3.0, implied_move=0.03, expected_move=0.01,
                        materiality=0.1, baseline_move=0.096))
    assert rich["setup"] != "VOL_SELL_RICH"
    assert "immaterial" in rich["reasons"]

    cheap = classify(row(move_ratio=0.2, implied_move=0.03, expected_move=0.14,
                         materiality=0.3, baseline_move=0.45))
    assert cheap["setup"] != "VOL_BUY_CHEAP"


def test_material_catalyst_still_trades():
    r = classify(row(move_ratio=2.0, implied_move=0.6, materiality=2.0))
    assert r["setup"] == "VOL_SELL_RICH"


def test_unknown_market_cap_is_not_treated_as_material():
    # Regression: EDGAR files no share count for 20-F filers, so market cap
    # came back null, expected_move fell back to the single-asset ceiling,
    # and mega-caps like AZN were flagged as cheap volatility.
    r = classify(row(materiality=float("nan"), market_cap=float("nan"),
                     move_ratio=0.13, implied_move=0.058, expected_move=0.45))
    assert r["setup"] not in ("VOL_BUY_CHEAP", "VOL_SELL_RICH")
    assert "size unknown" in r["reasons"]


def test_dilution_short_has_no_market_cap_gate():
    # The backtest showed the microcap requirement cancelled the runway
    # signal: runway<6mo alone is -5.4% @21d (p=0.000), but runway<6mo AND
    # microcap is -1.8% @21d (p=0.32).
    big = classify(row(runway_at_catalyst=-2.0, microcap=False))
    small = classify(row(runway_at_catalyst=-2.0, microcap=True))
    assert big["setup"] == "DILUTION_SHORT"
    assert small["setup"] == "DILUTION_SHORT"


def test_setups_carry_their_evidence_status():
    from biocatalyst.score import SETUP_EVIDENCE
    assert classify(row(runway_at_catalyst=-2.0))["evidence"] == "supported"
    assert SETUP_EVIDENCE["RUNUP_FADE"][0] == "not supported"
    assert SETUP_EVIDENCE["VOL_SELL_RICH"][0] == "untested"


def test_verdict_accompanies_every_setup():
    from biocatalyst.score import SETUPS, VERDICT
    assert set(VERDICT) == set(SETUPS)


def test_volatility_setups_are_not_labelled_directional():
    from biocatalyst.score import VERDICT
    assert VERDICT["VOL_SELL_RICH"][1] is False
    assert VERDICT["VOL_BUY_CHEAP"][1] is False
    assert VERDICT["DILUTION_SHORT"] == ("SHORT", True)
    assert VERDICT["BASE_RATE_LONG"] == ("LONG", True)


def test_classify_emits_a_verdict():
    r = classify(row(runway_at_catalyst=-2.0))
    assert r["verdict"] == "SHORT" and r["directional"] is True


def test_microcap_now_raises_short_conviction():
    # Corrected finding: the original size analysis mixed split-adjusted
    # prices with pre-split share counts and had the sign backwards.
    # Microcaps underperform (-10.5% @126d), so microcap should strengthen a
    # short rather than veto it.
    big = classify(row(runway_at_catalyst=-2.0, microcap=False))
    small = classify(row(runway_at_catalyst=-2.0, microcap=True))
    assert big["setup"] == small["setup"] == "DILUTION_SHORT"
    assert small["conviction"] > big["conviction"]


def test_heavy_filing_activity_raises_short_conviction():
    quiet = classify(row(runway_at_catalyst=-2.0, microcap=True, eightk_90d=2))
    busy = classify(row(runway_at_catalyst=-2.0, microcap=True, eightk_90d=8))
    assert busy["conviction"] > quiet["conviction"]
    assert "8-Ks in 90d" in busy["reasons"]


def test_filing_activity_only_amplifies_the_validated_setup():
    # A busy filer with plenty of cash must not become a signal.
    r = classify(row(runway_at_catalyst=30.0, eightk_90d=9, microcap=True))
    assert r["setup"] != "DILUTION_SHORT"
    assert "8-Ks in 90d" not in r["reasons"]
