"""Point-in-time financials.

Every number here is filtered by EDGAR's `filed` date, never by the period it
describes. A 10-Q covering Q3 is not knowable until it is filed weeks later,
and `companyfacts` happily serves restatements of old periods filed years
afterwards. Using the period end alone would leak the future into every
observation -- the exact failure that makes a backtest look good and then not
replicate.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ..config import DATA_DIR
from ..http import get
from ..sources.edgar import (BURN_TAGS, CASH_TAGS, INVESTMENT_TAGS,
                             SHARE_TAGS, ticker_to_cik)

CACHE = DATA_DIR / "cache"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
PERIODIC_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F"}


def companyfacts(ticker: str, refresh: bool = False) -> dict | None:
    """Fetch and disk-cache a company's XBRL facts. SEC throttles hard."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker.upper()}.json"
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    cik = ticker_to_cik().get(ticker.upper())
    if not cik:
        return None
    try:
        facts = get(FACTS_URL.format(cik=cik)).json().get("facts", {})
    except Exception:
        facts = {}
    path.write_text(json.dumps(facts))
    return facts


def _rows(facts: dict, tags: list[str], unit: str = "USD") -> list[dict]:
    out: list[dict] = []
    for tag in tags:
        node = facts.get("us-gaap", {}).get(tag) or facts.get("dei", {}).get(tag)
        if node:
            out.extend(node.get("units", {}).get(unit, []))
    return [r for r in out if r.get("filed") and r.get("val") is not None]


def _visible(rows: list[dict], asof: dt.date) -> list[dict]:
    """Only facts already filed on or before `asof`."""
    stamp = asof.isoformat()
    return [r for r in rows if r["filed"] <= stamp]


def _latest_by_period(rows: list[dict]):
    if not rows:
        return None, None, None
    best = max(rows, key=lambda r: (r.get("end", ""), r["filed"]))
    return best["val"], best.get("end"), best["filed"]


def _quarterly_burn(rows: list[dict]) -> float | None:
    """Average quarterly operating cash consumption from visible facts.

    Cash-flow facts in a 10-Q are cumulative year-to-date, so only spans of
    roughly one quarter are true quarterly figures.
    """
    qs = []
    for r in rows:
        if not (r.get("start") and r.get("end")):
            continue
        try:
            span = (dt.date.fromisoformat(r["end"])
                    - dt.date.fromisoformat(r["start"])).days
        except ValueError:
            continue
        if 60 <= span <= 100:
            qs.append(r)
    if not qs:
        return None
    qs.sort(key=lambda r: r["end"], reverse=True)
    recent = qs[:4]
    avg = sum(r["val"] for r in recent) / len(recent)
    return -avg if avg < 0 else 0.0


def pit_financials(ticker: str, asof: dt.date, facts: dict | None = None) -> dict | None:
    """Cash, burn and runway exactly as they were knowable on `asof`."""
    facts = facts if facts is not None else companyfacts(ticker)
    if not facts:
        return None

    cash_rows = _visible(_rows(facts, CASH_TAGS), asof)
    inv_rows = _visible(_rows(facts, INVESTMENT_TAGS), asof)
    burn_rows = _visible(_rows(facts, BURN_TAGS), asof)
    share_rows = _visible(_rows(facts, SHARE_TAGS, unit="shares"), asof)

    cash, cash_end, cash_filed = _latest_by_period(cash_rows)
    inv, _, _ = _latest_by_period(inv_rows)
    shares, _, _ = _latest_by_period(share_rows)
    if cash is None:
        return None

    total_cash = cash + (inv or 0)
    burn_q = _quarterly_burn(burn_rows)
    runway = None
    if burn_q and burn_q > 0:
        runway = round(total_cash / (burn_q / 3.0), 2)

    return {
        "ticker": ticker.upper(),
        "asof": asof,
        "cash_usd": float(total_cash),
        "cash_period_end": cash_end,
        "cash_filed": cash_filed,
        "quarterly_burn_usd": burn_q,
        "runway_months": runway,
        "shares_outstanding": float(shares) if shares else None,
    }


def filing_dates(ticker: str, start: dt.date, end: dt.date,
                 facts: dict | None = None) -> list[dt.date]:
    """Distinct periodic-report filing dates in the window.

    Filing dates are immutable and public the day they happen, which makes
    them the one event anchor in this project with no revision risk.
    """
    facts = facts if facts is not None else companyfacts(ticker)
    if not facts:
        return []
    seen = set()
    for r in _rows(facts, CASH_TAGS):
        if r.get("form") not in PERIODIC_FORMS:
            continue
        try:
            d = dt.date.fromisoformat(r["filed"])
        except (ValueError, KeyError):
            continue
        if start <= d <= end:
            seen.add(d)
    return sorted(seen)


SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SHARD = "https://data.sec.gov/submissions/{name}"


def all_filings(ticker: str, refresh: bool = False) -> list[tuple[str, str]]:
    """Every (form, filing-date) pair for a company, disk-cached.

    One fetch serves 8-K cadence, shelf registrations, offering prospectuses
    and Form 4 counts -- they all live in the same submissions index.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker.upper()}.forms.json"
    if path.exists() and not refresh:
        try:
            return [tuple(x) for x in json.loads(path.read_text())]
        except Exception:
            pass
    cik = ticker_to_cik().get(ticker.upper())
    if not cik:
        return []
    try:
        payload = get(SUBMISSIONS.format(cik=cik)).json()
    except Exception:
        return []
    recent = payload.get("filings", {}).get("recent", {})
    out = [(str(f), str(d)) for f, d in zip(recent.get("form", []),
                                            recent.get("filingDate", []))]
    try:
        path.write_text(json.dumps(out))
    except OSError:
        pass
    return out


def submissions(ticker: str, since: dt.date | None = None,
                refresh: bool = False) -> list[dt.date]:
    """Every 8-K filing date for a company, disk-cached.

    Filing dates are the cleanest anchor in this project: a filing dated
    2023-05-05 was public that day and is never revised. The `recent` array
    holds the last 1000 filings, which for most biotechs reaches back a decade
    -- but a heavy filer can exhaust it in a couple of years, so older shards
    are pulled when the window demands it.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker.upper()}.8k.json"
    if path.exists() and not refresh:
        try:
            cached = json.loads(path.read_text())
            dates = [dt.date.fromisoformat(d) for d in cached]
            if not since or (dates and min(dates) <= since):
                return dates
        except Exception:
            pass

    cik = ticker_to_cik().get(ticker.upper())
    if not cik:
        return []
    try:
        payload = get(SUBMISSIONS.format(cik=cik)).json()
    except Exception:
        return []

    def _collect(block: dict) -> list[dt.date]:
        out = []
        for form, filed in zip(block.get("form", []),
                               block.get("filingDate", [])):
            if not str(form).startswith("8-K"):
                continue
            try:
                out.append(dt.date.fromisoformat(filed))
            except ValueError:
                continue
        return out

    recent = payload.get("filings", {}).get("recent", {})
    dates = _collect(recent)
    all_dates = recent.get("filingDate", [])
    earliest = min(all_dates) if all_dates else None

    # Only reach for older shards when `recent` does not span the window.
    if since and earliest and dt.date.fromisoformat(earliest) > since:
        for shard in payload.get("filings", {}).get("files", []):
            try:
                dates.extend(_collect(get(SHARD.format(name=shard["name"])).json()))
            except Exception:
                continue

    dates = sorted(set(dates))
    path.write_text(json.dumps([d.isoformat() for d in dates]))
    return dates


def cadence_at(dates: list[dt.date], asof: dt.date) -> dict:
    """8-K activity as of a date, using only filings already made.

    `burst` is the 90-day count against the company's own trailing-year rate.
    A raw count mostly measures company size and filing habit; the ratio asks
    whether this company is unusually busy *for itself*.
    """
    visible = [d for d in dates if d <= asof]
    if not visible:
        return {"eightk_30d": 0, "eightk_90d": 0, "eightk_365d": 0,
                "days_since_8k": None, "burst": None}

    n30 = sum(1 for d in visible if 0 <= (asof - d).days <= 30)
    n90 = sum(1 for d in visible if 0 <= (asof - d).days <= 90)
    n365 = sum(1 for d in visible if 0 <= (asof - d).days <= 365)

    # Needs a year of history before a ratio means anything.
    expected = n365 / 4.0
    burst = round(n90 / expected, 3) if expected >= 0.75 else None

    return {"eightk_30d": n30, "eightk_90d": n90, "eightk_365d": n365,
            "days_since_8k": (asof - max(visible)).days, "burst": burst}


_SPLIT_CACHE: dict[str, list[tuple[dt.date, float]]] = {}


def split_history(ticker: str) -> list[tuple[dt.date, float]]:
    """Split events as (date, ratio); a 1-for-10 reverse split has ratio 0.1."""
    key = ticker.upper()
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    out: list[tuple[dt.date, float]] = []
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import yfinance as yf
        s = yf.Ticker(key).splits
        for stamp, ratio in s.items():
            if ratio and ratio > 0:
                out.append((stamp.date(), float(ratio)))
    except Exception:
        out = []
    _SPLIT_CACHE[key] = out
    return out


def shares_in_current_terms(shares: float | None, ticker: str,
                            asof: dt.date) -> float | None:
    """Restate a historical share count on today's post-split basis.

    Price history is back-adjusted for splits, so a price from 2023 is quoted
    in today's share terms while EDGAR's share count for 2023 is not.
    Multiplying the two directly inflated market caps by the cumulative split
    factor -- which for serial reverse-splitters put distressed nano-caps at
    trillion-dollar valuations. Restating the share count makes the two sides
    consistent.
    """
    if shares is None:
        return None
    factor = 1.0
    for when, ratio in split_history(ticker):
        if when > asof:
            factor *= ratio
    return shares * factor


def market_cap_at(ticker: str, asof: dt.date, price: float,
                  facts: dict | None = None) -> float | None:
    """Market cap on a past date, consistent with back-adjusted prices."""
    fin = pit_financials(ticker, asof, facts=facts)
    if not fin:
        return None
    shares = shares_in_current_terms(fin.get("shares_outstanding"),
                                     ticker, asof)
    return shares * price if shares else None


def short_interest_history(ticker: str, refresh: bool = False) -> list[dict]:
    """Full FINRA short-interest series for a ticker, disk-cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker.upper()}.si.json"
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    from ..sources import shortinterest
    df = shortinterest.fetch_ticker(ticker)
    rows = []
    if not df.empty:
        for _, r in df.iterrows():
            rows.append({
                "settlement_date": r["settlement_date"].isoformat(),
                "published_date": r["published_date"].isoformat(),
                "shares_short": r["shares_short"],
                "shares_short_prior": r["shares_short_prior"],
                "days_to_cover": r["days_to_cover"],
            })
    try:
        path.write_text(json.dumps(rows))
    except OSError:
        pass
    return rows


def short_interest_at(rows: list[dict], asof: dt.date) -> dict | None:
    """The most recent reading that was *published* on or before `asof`.

    Keying off settlement date instead would hand the study a position that
    the market could not see for another eight business days.
    """
    stamp = asof.isoformat()
    visible = [r for r in rows if r.get("published_date", "") <= stamp]
    if not visible:
        return None
    return max(visible, key=lambda r: r["published_date"])


_INSIDER_PANEL = {}


def insider_panel(quarters: list[str], tickers: set[str]):
    """Open-market insider transactions across several quarters, cached.

    Keyed by FILING date. Form 4 is due within two business days of the
    trade, but the filing is when the market could see it, so that is the
    only honest key for a point-in-time study.
    """
    import pandas as pd
    from ..sources import insider as _ins

    key = (tuple(quarters), len(tickers))
    if key in _INSIDER_PANEL:
        return _INSIDER_PANEL[key]

    frames = []
    for q in quarters:
        df = _ins.load_bulk_quarter(q)
        if df.empty:
            continue
        frames.append(df[df["ticker"].isin(tickers)])
    panel = (pd.concat(frames, ignore_index=True)
             if frames else pd.DataFrame())
    if not panel.empty:
        panel = panel.sort_values("filed_date")
    _INSIDER_PANEL[key] = panel
    return panel


def insider_at(panel, ticker: str, asof: dt.date, window: int = 90) -> dict:
    """Trailing-window insider activity visible on `asof`."""
    if panel is None or len(panel) == 0:
        return {}
    lo = asof - dt.timedelta(days=window)
    sub = panel[(panel["ticker"] == ticker)
                & (panel["filed_date"] > lo)
                & (panel["filed_date"] <= asof)]
    if len(sub) == 0:
        return {"insider_net_usd": 0.0, "insider_buy_usd": 0.0,
                "insider_sell_usd": 0.0, "insider_filings": 0,
                "insider_buyers": 0, "any_insider_activity": False}

    buys = sub[sub["code"] == "P"]
    sells = sub[sub["code"] == "S"]
    buy_usd = float(buys["usd"].sum())
    sell_usd = float(sells["usd"].sum())
    return {
        "insider_net_usd": buy_usd - sell_usd,
        "insider_buy_usd": buy_usd,
        "insider_sell_usd": sell_usd,
        "insider_filings": int(sub["accession"].nunique()),
        "insider_buyers": int(buys["accession"].nunique()),
        "any_insider_activity": True,
    }
