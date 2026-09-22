"""Join the raw tables into one catalyst-level feature frame."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import baserates
from .sources import shortinterest

# How much to trust a catalyst date, by how precisely BPC states it.
PRECISION_WEIGHT = {
    "day": 1.0, "month_part": 0.8, "month": 0.6,
    "quarter": 0.4, "half": 0.25, "year": 0.15, "unknown": 0.0,
}

JOINED_SQL = """
SELECT c.*,
       t.overall_status, t.phases, t.enrollment, t.allocation, t.masking,
       t.primary_completion, t.lead_sponsor,
       f.cash_usd, f.quarterly_burn_usd, f.runway_months, f.shares_outstanding,
       p.last, p.ret_20d, p.ret_60d, p.rvol_20d, p.rvol_60d,
       p.adv_usd, p.pct_of_52w_high, p.market_cap,
       i.atm_iv, i.implied_move, i.expiry AS implied_expiry,
       s.sentiment, s.thin AS sentiment_thin, s.articles AS news_articles,
       s.scored_articles, s.top_positive, s.top_negative,
       s.eightk_30d, s.eightk_90d, s.days_since_8k,
       si.shares_short, si.shares_short_prior, si.days_to_cover,
       si.settlement_date AS short_asof,
       fz.dilution_label, fz.dilution_readiness, fz.offerings_24m,
       fz.days_since_offering, fz.shelf_live,
       ins.insider_tilt, ins.net_usd AS insider_net_usd,
       ins.form4_filings
FROM catalysts c
LEFT JOIN trials     t ON c.nct_id = t.nct_id
LEFT JOIN financials f ON c.ticker = f.ticker
LEFT JOIN prices     p ON c.ticker = p.ticker
LEFT JOIN implied    i ON c.ticker = i.ticker
                      AND c.catalyst_date_lo = i.catalyst_date
-- Most recent sentiment snapshot only; the rest of the log is for backtesting.
LEFT JOIN (
    SELECT * FROM sentiment
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_date DESC) = 1
) s ON c.ticker = s.ticker
-- Latest short-interest reading only; the rest of the series is for backtesting.
LEFT JOIN (
    SELECT * FROM short_interest
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY settlement_date DESC) = 1
) si ON c.ticker = si.ticker
LEFT JOIN financing fz ON c.ticker = fz.ticker
LEFT JOIN (
    SELECT * FROM insider
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_date DESC) = 1
) ins ON c.ticker = ins.ticker
"""


def build(con, horizon_days: int = 180, today: dt.date | None = None) -> pd.DataFrame:
    """Return upcoming catalysts enriched with every derived feature."""
    today = today or dt.date.today()
    df = con.execute(JOINED_SQL).fetchdf()
    if df.empty:
        return df

    df["catalyst_date_lo"] = pd.to_datetime(df["catalyst_date_lo"]).dt.date
    df["catalyst_date_hi"] = pd.to_datetime(df["catalyst_date_hi"]).dt.date

    df["days_to_catalyst"] = df["catalyst_date_lo"].map(
        lambda d: (d - today).days if pd.notna(d) else None)
    df["date_confidence"] = df["date_precision"].map(PRECISION_WEIGHT).fillna(0.0)

    # Forward-looking window only. Catalysts stated as a range are still live
    # until the far end of that range has passed.
    upcoming = df["catalyst_date_hi"].map(
        lambda d: pd.notna(d) and d >= today)
    in_window = df["days_to_catalyst"].map(
        lambda d: d is not None and d <= horizon_days)
    df = df[upcoming & in_window].copy()
    if df.empty:
        return df

    df["ta"] = df["indication"].map(baserates.classify_ta)
    df["loa"] = df.apply(
        lambda r: baserates.loa(r["simplified_stage"], r["indication"],
                                r["designations"]), axis=1)
    df["expected_move"] = df.apply(
        lambda r: baserates.typical_catalyst_move(
            r["simplified_stage"], r["market_cap"]), axis=1)

    # Market's price for the event vs. what that event type usually delivers.
    df["move_ratio"] = df["implied_move"] / df["expected_move"]

    # Materiality: is this catalyst even big enough to show up above the
    # stock's ordinary noise over the same horizon? A Phase 3 readout worth
    # ~1% of a mega-cap is swamped by its baseline vol, so any comparison of
    # straddle price to single-asset value-at-risk is a category error there.
    horizon_yrs = (df["days_to_catalyst"].clip(lower=1)) / 252.0
    df["baseline_move"] = df["rvol_60d"] * (horizon_yrs ** 0.5)
    df["materiality"] = df["expected_move"] / df["baseline_move"]

    # Without a market cap the expected move falls back to the single-asset
    # ceiling, which would label a foreign-listed mega-cap as material. EDGAR
    # carries no share count for 20-F filers, so this is common. Unknown size
    # means unknown materiality, not small.
    df.loc[df["market_cap"].isna(), "materiality"] = float("nan")

    # Months of cash left once the catalyst has come and gone.
    df["runway_at_catalyst"] = df["runway_months"] - (df["days_to_catalyst"] / 30.44)

    squeeze = df.apply(
        lambda r: shortinterest.squeeze_metrics(
            r.get("shares_short"), r.get("days_to_cover"),
            r.get("shares_outstanding"), r.get("shares_short_prior")),
        axis=1, result_type="expand")
    df = pd.concat([df, squeeze], axis=1)

    df["is_binary"] = df["simplified_stage"].isin(
        ["phase2", "phase2/3", "phase3", "pdufa", "nda", "bla"])
    df["microcap"] = df["market_cap"].fillna(0) < 300e6
    df["illiquid"] = df["adv_usd"].fillna(0) < 1e6
    return df
