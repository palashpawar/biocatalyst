"""Shelf registrations and offerings — how ready a company is to dilute.

This exists to sharpen the one validated setup rather than add a new one. The
dilution thesis says a company with months of cash left will have to raise;
an effective shelf says it can price a deal overnight, and a long history of
424B5 prospectuses says it habitually does.

All of it comes from the EDGAR submissions index, which is already cached for
8-K cadence, so it costs no extra requests. Filing dates are immutable, so
every figure here is point-in-time by construction.
"""
from __future__ import annotations

import datetime as dt

# A shelf registration goes stale after three years under Rule 415.
SHELF_LIFE_DAYS = 365 * 3

SHELF_FORMS = ("S-3", "S-3ASR", "S-3/A", "S-3MEF")
OFFERING_FORMS = ("424B5", "424B4", "424B3")


def _dates(filings: list[tuple[str, str]], prefixes: tuple[str, ...],
           asof: dt.date) -> list[dt.date]:
    out = []
    for form, filed in filings:
        if not any(form == p or form.startswith(p) for p in prefixes):
            continue
        try:
            d = dt.date.fromisoformat(filed)
        except ValueError:
            continue
        if d <= asof:
            out.append(d)
    return sorted(out, reverse=True)


def financing_at(filings: list[tuple[str, str]],
                 asof: dt.date | None = None) -> dict:
    """Shelf and offering posture as of a date."""
    asof = asof or dt.date.today()
    shelves = _dates(filings, SHELF_FORMS, asof)
    offerings = _dates(filings, OFFERING_FORMS, asof)

    last_shelf = shelves[0] if shelves else None
    shelf_age = (asof - last_shelf).days if last_shelf else None
    # Treated as live while inside the three-year window. This is an
    # approximation: effectiveness is not in the index, and a shelf can be
    # exhausted or withdrawn without another filing.
    shelf_live = shelf_age is not None and shelf_age <= SHELF_LIFE_DAYS

    last_offering = offerings[0] if offerings else None
    return {
        "last_shelf_date": last_shelf,
        "shelf_age_days": shelf_age,
        "shelf_live": shelf_live,
        "shelf_count": len(shelves),
        "last_offering_date": last_offering,
        "days_since_offering": ((asof - last_offering).days
                                if last_offering else None),
        "offerings_24m": sum(1 for d in offerings
                             if (asof - d).days <= 730),
        "offerings_total": len(offerings),
    }


def dilution_readiness(fin: dict, runway_months: float | None) -> dict:
    """How loaded the financing gun is.

    A serial issuer with a live shelf and little cash is the setup the runway
    backtest was picking up; this names the mechanism rather than inferring it
    from the cash balance alone.
    """
    score = 0
    notes = []
    if fin.get("shelf_live"):
        score += 1
        notes.append("live shelf")
    if (fin.get("offerings_24m") or 0) >= 2:
        score += 1
        notes.append(f"{fin['offerings_24m']} offerings in 24mo")

    # Readiness is ability multiplied by need. A funded company can have a
    # shelf and a long 424B5 history and still not be a dilution risk -- for
    # a large pharma those prospectuses are usually bond issues, not equity.
    # Without the need, capability alone tops out at "possible".
    needs_cash = runway_months is not None and runway_months < 24
    if runway_months is not None and runway_months < 12:
        score += 1
    if not needs_cash:
        score = min(score, 1)
        if fin.get("shelf_live"):
            notes = ["live shelf, but funded"]

    days = fin.get("days_since_offering")
    if days is not None and days <= 120 and needs_cash:
        notes.append(f"priced a deal {days}d ago")

    label = {0: "cold", 1: "possible", 2: "likely", 3: "loaded"}[min(score, 3)]
    return {"dilution_readiness": score, "dilution_label": label,
            "dilution_notes": "; ".join(notes)}
