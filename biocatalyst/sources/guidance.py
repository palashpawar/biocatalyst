"""Company-stated readout timing, from 8-K full text.

A ClinicalTrials.gov primary completion date is when the last patient reaches
the endpoint, not when the company says anything. Measured across 2,190
day-precision readouts, only 3.3% of the largest abnormal moves landed within
five days of that date -- below the 5.2% chance would produce. So the calendar
treats those dates as the opening of a window rather than an event.

This module narrows that window using what the company itself has guided to:
"topline data expected in the first half of 2027" is a real statement about
timing, made on a dated filing, in a way a completion date is not.

The guidance is company-level, not trial-level. A sponsor running several
programmes may guide to one of them, so the window is *intersected* with the
trial's own rather than replaced by it -- narrowing only where the two agree,
and never moving a catalyst outside the range the trial supports.
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from ..http import get
from .bpc import parse_catalyst_date
from .discovery import _clean_filing

EFTS = "https://efts.sec.gov/LATEST/search-index"

# Several phrasings, because companies do not agree on one.
QUERIES = [
    '"topline data expected"',
    '"topline results expected"',
    '"data expected in"',
    '"results are expected"',
    '"readout expected"',
    '"expect to report topline"',
]

_PERIOD = (
    r"((?:the\s+)?(?:first|second|third|fourth)\s+(?:half|quarter)\s+of\s+\d{4}"
    r"|Q[1-4]\s*(?:of\s*)?\d{4}"
    r"|H[12]\s*\d{4}"
    r"|(?:mid|early|late)[-\s]\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4})"
)

GUIDANCE_RE = re.compile(
    r"(?:topline|top-line|data|results|readout)\s+(?:are\s+|is\s+)?"
    r"(?:expected|anticipated|projected|on\s+track)\s+"
    r"(?:to\s+be\s+(?:reported|announced|available|released)\s+)?"
    r"(?:in|by|during|for)?\s*" + _PERIOD, re.I)

_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_period(text: str) -> tuple[dt.date | None, dt.date | None]:
    """Turn a guided period into a date range."""
    t = re.sub(r"\s+", " ", text).strip().lower()
    year = re.search(r"(19|20)\d{2}", t)
    if not year:
        return None, None
    y = int(year.group(0))

    m = re.search(r"(first|second|third|fourth)\s+(half|quarter)", t)
    if m:
        n, unit = _ORDINAL[m.group(1)], m.group(2)
        if unit == "half":
            return ((dt.date(y, 1, 1), dt.date(y, 6, 30)) if n == 1
                    else (dt.date(y, 7, 1), dt.date(y, 12, 31)))
        lo_m, hi_m = 3 * n - 2, 3 * n
        import calendar
        return dt.date(y, lo_m, 1), dt.date(y, hi_m,
                                            calendar.monthrange(y, hi_m)[1])

    # Q1 2027 / H2 2027 / mid-2027 / March 2027 all parse with the existing
    # fuzzy date parser, which already understands these shapes.
    lo, hi, _ = parse_catalyst_date(t)
    if lo:
        return lo, hi
    if re.search(r"\bmid[-\s]", t):
        return dt.date(y, 5, 1), dt.date(y, 8, 31)
    if re.search(r"\bearly\b", t):
        return dt.date(y, 1, 1), dt.date(y, 4, 30)
    if re.search(r"\blate\b", t):
        return dt.date(y, 9, 1), dt.date(y, 12, 31)
    return None, None


def _ticker_from(names: list) -> str | None:
    if not names:
        return None
    m = re.search(r"\(([A-Z][A-Z0-9.\-]{0,5})(?:,|\))", names[0])
    return m.group(1) if m else None


def fetch(months_back: int = 9, per_query: int = 100,
          verbose: bool = False) -> pd.DataFrame:
    """Recent 8-Ks containing readout guidance, with the period parsed out."""
    end = dt.date.today()
    start = end - dt.timedelta(days=30 * months_back)
    seen, rows = set(), []

    for q in QUERIES:
        offset = 0
        while offset < per_query:
            try:
                payload = get(EFTS, params={
                    "q": q, "forms": "8-K",
                    "startdt": start.isoformat(), "enddt": end.isoformat(),
                    "from": offset}).json()
            except Exception:
                break
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                doc_id = h.get("_id", "")
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                ticker = _ticker_from(src.get("display_names") or [])
                cik = (src.get("ciks") or [None])[0]
                if not ticker or not cik:
                    continue
                rows.append({"ticker": ticker, "cik": cik, "doc_id": doc_id,
                             "filed": src.get("file_date")})
            offset += len(hits)
            if len(hits) < 10:
                break
        if verbose:
            print(f"    guidance {q}: {len(rows)} filings so far", flush=True)

    out = []
    for r in rows:
        adsh, _, fname = r["doc_id"].partition(":")
        if not adsh or not fname:
            continue
        try:
            url = (f"https://www.sec.gov/Archives/edgar/data/{int(r['cik'])}/"
                   f"{adsh.replace('-', '')}/{fname}")
            text = _clean_filing(get(url).text)
        except Exception:
            continue
        m = GUIDANCE_RE.search(text)
        if not m:
            continue
        lo, hi = parse_period(m.group(1))
        if not lo:
            continue
        out.append({
            "ticker": r["ticker"].upper(),
            "filed": dt.date.fromisoformat(r["filed"]) if r["filed"] else None,
            "period_lo": lo,
            "period_hi": hi,
            "phrase": re.sub(r"\s+", " ", m.group(0))[:180],
            "pulled_at": dt.datetime.now(),
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df
    # Keep the most recent statement per company.
    return (df.sort_values("filed")
              .drop_duplicates(subset=["ticker"], keep="last")
              .reset_index(drop=True))


def _missing(v) -> bool:
    """None, NaN and NaT all mean 'no date'.

    `v is None` is not enough: a date column round-tripped through pandas
    comes back as NaT, which is not None and raises on comparison. Same trap
    as NaN being truthy.
    """
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def narrow(window_lo, window_hi, guide_lo, guide_hi):
    """Intersect a trial's readout window with company guidance.

    Returns (lo, hi, source). Guidance only ever narrows: if the two ranges do
    not overlap the trial window is kept, because a company guiding to a date
    the trial cannot support is more likely to be about a different programme
    than evidence the trial will read out early.
    """
    if _missing(guide_lo) or _missing(window_lo):
        return window_lo, window_hi, "trial"
    lo = max(window_lo, guide_lo)
    hi = (min(window_hi, guide_hi)
          if not _missing(guide_hi) and not _missing(window_hi)
          else window_hi)
    if _missing(hi) or lo > hi:
        return window_lo, window_hi, "trial"
    return lo, hi, "guidance"
