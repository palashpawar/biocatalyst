"""SEC EDGAR XBRL company facts -> cash, burn rate, and runway.

Runway is the single most actionable number in small-cap biotech: a company
with under two quarters of cash will finance, and financings are priced at a
discount into strength. This is public XBRL data, no key required.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from functools import lru_cache

import pandas as pd

from ..config import DATA_DIR
from ..http import get

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CASH_TAGS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]
INVESTMENT_TAGS = ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                   "AvailableForSaleSecuritiesDebtSecuritiesCurrent"]
BURN_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
SHARE_TAGS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]


@lru_cache(maxsize=1)
def ticker_to_cik() -> dict[str, int]:
    data = get(TICKERS_URL).json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def _units(facts: dict, tags: list[str], unit: str = "USD") -> list[dict]:
    for tag in tags:
        node = facts.get("us-gaap", {}).get(tag) or facts.get("dei", {}).get(tag)
        if node:
            rows = node.get("units", {}).get(unit, [])
            if rows:
                return rows
    return []


def _latest(rows: list[dict]):
    dated = [r for r in rows if r.get("end") and r.get("val") is not None]
    if not dated:
        return None, None
    best = max(dated, key=lambda r: r["end"])
    return best["val"], best["end"]


def _quarterly_burn(rows: list[dict]) -> float | None:
    """Average monthly operating cash burn from the most recent ~4 quarters."""
    qs = [r for r in rows
          if r.get("start") and r.get("end") and r.get("val") is not None
          and 60 <= (dt.date.fromisoformat(r["end"]) -
                     dt.date.fromisoformat(r["start"])).days <= 100]
    if not qs:
        return None
    qs.sort(key=lambda r: r["end"], reverse=True)
    recent = qs[:4]
    avg = sum(r["val"] for r in recent) / len(recent)
    return -avg if avg < 0 else 0.0  # positive number = cash consumed per quarter


CACHE_DIR = DATA_DIR / "cache"
CACHE_TTL_HOURS = 20


def _cached_facts(ticker: str, cik: int) -> dict | None:
    """Company facts with a short TTL.

    SEC throttles hard, and re-pulling 150+ filers on every refresh was the
    slowest leg of the pipeline by a wide margin. Financials only move when a
    10-Q lands, so a day-old copy is no worse than a fresh one -- and on CI a
    restored cache turns a ten-minute step into seconds.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker.upper()}.json"
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_HOURS:
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    try:
        facts = get(FACTS_URL.format(cik=cik)).json().get("facts", {})
    except Exception:
        # A stale copy beats no data when SEC is rate limiting.
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None
    try:
        path.write_text(json.dumps(facts))
    except OSError:
        pass
    return facts


def fetch_one(ticker: str) -> dict | None:
    cik = ticker_to_cik().get(ticker.upper())
    if not cik:
        return None
    facts = _cached_facts(ticker, cik)
    if not facts:
        return None

    cash, asof = _latest(_units(facts, CASH_TAGS))
    inv, _ = _latest(_units(facts, INVESTMENT_TAGS))
    total_cash = (cash or 0) + (inv or 0)
    burn_q = _quarterly_burn(_units(facts, BURN_TAGS))
    shares, _ = _latest(_units(facts, SHARE_TAGS, unit="shares"))

    runway = None
    if burn_q and burn_q > 0 and total_cash:
        runway = round(total_cash / (burn_q / 3.0), 1)

    return {
        "ticker": ticker.upper(),
        "cik": f"{cik:010d}",
        "cash_usd": float(total_cash) if total_cash else None,
        "cash_asof": dt.date.fromisoformat(asof) if asof else None,
        "quarterly_burn_usd": burn_q,
        "runway_months": runway,
        "shares_outstanding": float(shares) if shares else None,
        "pulled_at": dt.datetime.now(),
    }


def fetch(tickers: list[str], verbose: bool = False) -> pd.DataFrame:
    rows = []
    uniq = sorted({t.upper() for t in tickers if t})
    for i, t in enumerate(uniq, 1):
        rec = fetch_one(t)
        if rec:
            rows.append(rec)
        if verbose and i % 25 == 0:
            print(f"    EDGAR {i}/{len(uniq)}", flush=True)
    return pd.DataFrame(rows)
