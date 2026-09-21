"""BiopharmCatalyst FDA calendar.

Paywall policy
--------------
BPC's JSON endpoint returns fields the web UI renders as "Locked" for anonymous
visitors -- their proprietary likelihood-of-approval and cash-database numbers.
Those are the product they sell. This module whitelists the free-tier fields and
drops everything in PAID_FIELDS before the data reaches the database, so the
engine never depends on values we are not entitled to.

If you subscribe, set BIOCATALYST_BPC_COOKIE to your session cookie and flip
`include_paid=True`; the same parser will then keep those columns legitimately.
"""
from __future__ import annotations

import calendar
import datetime as dt
import os
import re

import pandas as pd

from ..http import get

API = "https://www.biopharmcatalyst.com/api/fda-calendar"

# Free-tier fields. Everything not listed here is discarded.
FREE_FIELDS = [
    "drug_id", "drug_name", "clinical_trial_id", "catalyst_date", "note",
    "indication", "company_name", "company_ticker", "company_price",
    "statuses", "simplified_stage", "stage_label", "fda_status_label",
    "next_catalyst_label", "company_optionable", "drug_updated_at",
]

# Subscriber-only. Never persisted unless include_paid is explicitly set.
PAID_FIELDS = [
    "likelihood_of_approval", "likelihood_of_progressing",
    "calculated_est_months_cash", "calculated_est_live_cash",
    "calculated_net_cash", "monthly_cash_burn_not_adjusted",
    "cash_equivalents_and_short_term_investments", "enterprise_value",
    "insider_holdings_pct", "shareinfo_float",
]

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})

_PART = {
    "early": (1, 10), "beginning": (1, 10), "start": (1, 10),
    "mid": (11, 20), "middle": (11, 20),
    "late": (21, 31), "end": (21, 31),
}


def _eom(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_catalyst_date(raw: str | None):
    """Parse BPC's fuzzy date strings into (lo, hi, precision).

    Handles '09/21/2026', 'Mid Sep 2026', 'September 2026', 'H1 2026',
    'Q4 2026', 'Early 2026' and bare years. Returns (None, None, 'unknown')
    when nothing parses, so unparseable rows survive rather than crash.
    """
    if not raw or not str(raw).strip():
        return None, None, "unknown"
    # Numeric formats are matched against the untouched string: normalising
    # hyphens to spaces first would silently destroy ISO dates like 2025-03-17.
    literal = str(raw).strip()
    text = literal.lower().replace("--", " ").replace("-", " ")

    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", literal)
    if m:
        mo, da, yr = (int(g) for g in m.groups())
        try:
            d = dt.date(yr, mo, da)
            return d, d, "day"
        except ValueError:
            return None, None, "unknown"

    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", literal)
    if m:
        yr, mo, da = (int(g) for g in m.groups())
        try:
            d = dt.date(yr, mo, da)
            return d, d, "day"
        except ValueError:
            return None, None, "unknown"

    # ISO year-month, e.g. ClinicalTrials.gov's "2026-11".
    m = re.fullmatch(r"((?:19|20)\d{2})-(\d{1,2})", literal)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (dt.date(year, month, 1),
                    dt.date(year, month, _eom(year, month)), "month")

    # Month name with an explicit day: "November 15, 2026", "1 April 2027".
    # PDUFA dates in 8-K filings are written this way, and falling through to
    # the month-only branch below would silently round them to the 1st while
    # the caller still labelled them day-precision.
    names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    m = re.search(rf"\b({names})\b\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})",
                  text)
    if not m:
        m2 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({names})\b,?\s+((?:19|20)\d{{2}})",
                       text)
        if m2:
            day, month_name, yr = m2.group(1), m2.group(2), m2.group(3)
            m = None
        else:
            day = month_name = yr = None
    else:
        month_name, day, yr = m.group(1), m.group(2), m.group(3)
    if day and month_name and yr:
        try:
            d = dt.date(int(yr), _MONTHS[month_name], int(day))
            return d, d, "day"
        except ValueError:
            return None, None, "unknown"

    ym = re.search(r"(19|20)\d{2}", text)
    if not ym:
        return None, None, "unknown"
    year = int(ym.group(0))

    m = re.search(r"\bq([1-4])\b", text)
    if m:
        q = int(m.group(1))
        lo_m, hi_m = 3 * q - 2, 3 * q
        return dt.date(year, lo_m, 1), dt.date(year, hi_m, _eom(year, hi_m)), "quarter"

    m = re.search(r"\bh([12])\b|\b([12])h\b|\bfirst half\b|\bsecond half\b", text)
    if m:
        digit = m.group(1) or m.group(2)
        half = 1 if (digit == "1" or "first" in m.group(0)) else 2
        lo_m, hi_m = (1, 6) if half == 1 else (7, 12)
        return dt.date(year, lo_m, 1), dt.date(year, hi_m, _eom(year, hi_m)), "half"

    month = None
    for name, idx in _MONTHS.items():
        if re.search(rf"\b{name}\b", text):
            month = idx
            break

    if month:
        for word, (lo_d, hi_d) in _PART.items():
            if re.search(rf"\b{word}\b", text):
                hi_d = min(hi_d, _eom(year, month))
                return (dt.date(year, month, lo_d),
                        dt.date(year, month, hi_d), "month_part")
        return (dt.date(year, month, 1),
                dt.date(year, month, _eom(year, month)), "month")

    for word, _ in _PART.items():
        if re.search(rf"\b{word}\b", text):
            lo_m, hi_m = {"early": (1, 4), "beginning": (1, 4), "start": (1, 4),
                          "mid": (5, 8), "middle": (5, 8),
                          "late": (9, 12), "end": (9, 12)}[word]
            return (dt.date(year, lo_m, 1),
                    dt.date(year, hi_m, _eom(year, hi_m)), "year_part")

    return dt.date(year, 1, 1), dt.date(year, 12, 31), "year"


def _designations(statuses) -> str:
    if not isinstance(statuses, list):
        return ""
    tags = [s.get("abbreviation") for s in statuses if isinstance(s, dict)]
    return " ".join(sorted({t for t in tags if t}))


def fetch(max_pages: int | None = None, include_paid: bool = False) -> pd.DataFrame:
    """Pull the full FDA calendar. Returns one row per drug/catalyst."""
    cookies = {}
    if include_paid:
        cookie = os.environ.get("BIOCATALYST_BPC_COOKIE")
        if not cookie:
            raise RuntimeError(
                "include_paid=True requires BIOCATALYST_BPC_COOKIE (your own "
                "subscriber session). Refusing to pull paid fields anonymously."
            )
        cookies = {"bpc_session": cookie}

    rows, page, last_page = [], 1, 1
    while True:
        payload = get(API, params={"page": page},
                      headers={"Accept": "application/json"},
                      cookies=cookies).json()
        rows.extend(payload.get("data", []))
        last_page = payload.get("meta", {}).get("last_page", page)
        if page >= last_page or (max_pages and page >= max_pages):
            break
        page += 1

    keep = FREE_FIELDS + (PAID_FIELDS if include_paid else [])
    clean = []
    for r in rows:
        row = {k: r.get(k) for k in keep}
        row["designations"] = _designations(r.get("statuses"))
        lo, hi, prec = parse_catalyst_date(r.get("catalyst_date"))
        row["catalyst_date_lo"], row["catalyst_date_hi"] = lo, hi
        row["date_precision"] = prec
        clean.append(row)

    df = pd.DataFrame(clean)
    if df.empty:
        return df
    df = df.drop(columns=["statuses"], errors="ignore")
    df = df.rename(columns={
        "company_ticker": "ticker", "clinical_trial_id": "nct_id",
        "stage_label": "stage", "fda_status_label": "fda_status",
        "next_catalyst_label": "next_catalyst", "catalyst_date": "catalyst_date_raw",
        "company_optionable": "optionable", "company_price": "price",
    })
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["optionable"] = df["optionable"].astype("boolean")
    df["pulled_at"] = dt.datetime.now()
    df = df.drop(columns=["drug_updated_at"], errors="ignore")

    # The API pads pages with placeholder rows that carry a sentinel date and
    # no stage/status/catalyst label. They are not real catalysts.
    placeholder = df["stage"].isna() & df["next_catalyst"].isna() & df["fda_status"].isna()
    df = df[~placeholder]

    return df.drop_duplicates(subset=["drug_id", "catalyst_date_raw"])
