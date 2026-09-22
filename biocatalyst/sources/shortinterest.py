"""FINRA consolidated short interest — squeeze risk.

Why this source: FINRA publishes it free, without auth, and with history back
to 2017. That makes short interest the one risk input here that can actually
go through the backtest harness, unlike headline sentiment.

Two timing facts shape everything below:

  * It is **bi-monthly**, settled mid-month and month-end. A reading is up to
    two weeks stale before the next one lands.
  * It is **published about eight business days after settlement**. A position
    measured on the 31st is not public until roughly the 12th of the next
    month, so any historical use must key off the publication date, never the
    settlement date, or the study quietly sees the future.
"""
from __future__ import annotations

import csv
import datetime as dt
import io

import pandas as pd

from ..http import post

API = ("https://api.finra.org/data/group/otcMarket/name/"
       "consolidatedShortInterest")

# FINRA's dissemination lag. Eight business days is their published schedule;
# ten is used here so a holiday week cannot make a reading look knowable
# earlier than it was.
PUBLICATION_LAG_BDAYS = 10


def publication_date(settlement: dt.date) -> dt.date:
    """When a settlement date's figures actually became public."""
    return (pd.Timestamp(settlement)
            + pd.tseries.offsets.BDay(PUBLICATION_LAG_BDAYS)).date()


def _parse(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _num(v):
    try:
        return float(v) if v not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


def fetch_ticker(ticker: str, limit: int = 5000) -> pd.DataFrame:
    """Full short-interest history for one symbol."""
    try:
        resp = post(API, json={
            "limit": limit,
            "compareFilters": [{"fieldName": "symbolCode",
                                "fieldValue": ticker.upper(),
                                "compareType": "EQUAL"}],
        })
        rows = _parse(resp.text)
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()

    out = []
    for r in rows:
        try:
            settle = dt.date.fromisoformat(r["settlementDate"])
        except (KeyError, ValueError):
            continue
        shares_short = _num(r.get("currentShortPositionQuantity"))
        if shares_short is None:
            continue
        out.append({
            "ticker": ticker.upper(),
            "settlement_date": settle,
            "published_date": publication_date(settle),
            "shares_short": shares_short,
            "shares_short_prior": _num(r.get("previousShortPositionQuantity")),
            "avg_daily_volume": _num(r.get("averageDailyVolumeQuantity")),
            "days_to_cover": _num(r.get("daysToCoverQuantity")),
            "change_pct": _num(r.get("changePercent")),
            "pulled_at": dt.datetime.now(),
        })
    df = pd.DataFrame(out)
    return df.sort_values("settlement_date") if not df.empty else df


def latest(ticker: str) -> dict | None:
    df = fetch_ticker(ticker)
    return None if df.empty else df.iloc[-1].to_dict()


def fetch(tickers: list[str], history: bool = False,
          verbose: bool = False) -> pd.DataFrame:
    """Short interest for a list of tickers, latest reading or full history."""
    frames = []
    uniq = sorted({t.upper() for t in tickers if t})
    for i, t in enumerate(uniq, 1):
        df = fetch_ticker(t)
        if not df.empty:
            frames.append(df if history else df.tail(1))
        if verbose and i % 25 == 0:
            print(f"    short interest {i}/{len(uniq)}", flush=True)
    return (pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame())


def squeeze_metrics(shares_short: float | None,
                    days_to_cover: float | None,
                    shares_outstanding: float | None,
                    prior_short: float | None = None) -> dict:
    """Crowding measures for the short side of a position.

    Days-to-cover is the metric that matters: it is how many normal sessions
    of volume the shorts would need to get out, and a crowded exit is what
    turns a bad print into a squeeze. Percent of shares outstanding is
    reported as exactly that -- float is not available free, and calling an
    outstanding-share ratio "percent of float" would overstate nothing but
    understate the true crowding.
    """
    pct_out = None
    if shares_short and shares_outstanding:
        pct_out = shares_short / shares_outstanding

    build = None
    if shares_short and prior_short:
        build = shares_short / prior_short - 1

    # Tiered rather than continuous: the underlying data is bi-monthly and
    # stale, so fine gradations would be false precision.
    risk = 0
    if days_to_cover is not None:
        if days_to_cover >= 10:
            risk = 3
        elif days_to_cover >= 5:
            risk = 2
        elif days_to_cover >= 2:
            risk = 1
    if pct_out is not None and pct_out >= 0.20:
        risk = max(risk, 2)
    if pct_out is not None and pct_out >= 0.30:
        risk = 3

    label = {0: "low", 1: "moderate", 2: "elevated", 3: "high"}[risk]
    return {"short_pct_outstanding": pct_out, "short_build": build,
            "squeeze_risk": risk, "squeeze_label": label}
