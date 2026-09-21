"""Historical event universes.

Two anchors, with very different reliability:

`filing_events`   -- 10-Q/10-K filing dates. Immutable, public the day they
                     happen. The clean anchor.
`catalyst_events` -- ClinicalTrials.gov primary completion dates. The API
                     serves only the *current* value of that date, and there
                     is no public version history (the v2 history endpoint is
                     404 and the internal one 403). Sponsors revise these
                     dates, often repeatedly, so a date used here may not have
                     been knowable beforehand. Results on this anchor are
                     optimistic about timing and are labelled as such.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ..http import get
from ..sources.bpc import parse_catalyst_date
from ..sources.discovery import _simplify_stage, sponsor_to_ticker
from . import pit

CTGOV = "https://clinicaltrials.gov/api/v2/studies"


def catalyst_events(start: dt.date, end: dt.date,
                    phases=("PHASE2", "PHASE3"),
                    max_studies: int = 6000,
                    verbose: bool = False) -> pd.DataFrame:
    """Industry-sponsored trials with primary completion inside the window."""
    rows, token = [], None
    while len(rows) < max_studies:
        params = {
            "query.term": "AREA[LeadSponsorClass]INDUSTRY",
            "filter.advanced": f"AREA[PrimaryCompletionDate]RANGE[{start},{end}]",
            "fields": ("NCTId,Phase,PrimaryCompletionDate,LeadSponsorName,"
                       "EnrollmentCount,Condition,OverallStatus,BriefTitle"),
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        payload = get(CTGOV, params=params).json()
        studies = payload.get("studies", [])
        if not studies:
            break
        for s in studies:
            ps = s.get("protocolSection", {})
            design = ps.get("designModule", {})
            ph = design.get("phases") or []
            if phases and not any(p in phases for p in ph):
                continue
            conds = ps.get("conditionsModule", {}).get("conditions") or []
            raw = (ps["statusModule"].get("primaryCompletionDateStruct", {})
                   .get("date"))
            lo, _, prec = parse_catalyst_date(raw)
            if not lo:
                continue
            rows.append({
                "nct_id": ps["identificationModule"]["nctId"],
                "drug_name": ps["identificationModule"].get("briefTitle"),
                "event_date": lo,
                "date_precision": prec,
                "stage_raw": ",".join(ph),
                "simplified_stage": _simplify_stage(",".join(ph)),
                "indication": "; ".join(conds[:3]),
                "lead_sponsor": ps["sponsorCollaboratorsModule"]["leadSponsor"]["name"],
                "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
                "overall_status": ps["statusModule"].get("overallStatus"),
            })
        token = payload.get("nextPageToken")
        if verbose:
            print(f"    ctgov {len(rows)} events", flush=True)
        if not token:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ticker"] = df["lead_sponsor"].map(sponsor_to_ticker)
    df = df[df["ticker"].notna()].copy()
    df["anchor"] = "ctgov"
    return df.drop_duplicates(subset=["nct_id"])


def filing_events(tickers: list[str], start: dt.date, end: dt.date,
                  verbose: bool = False) -> pd.DataFrame:
    """One observation per periodic filing: the clean point-in-time anchor."""
    rows = []
    uniq = sorted({t.upper() for t in tickers if t})
    for i, t in enumerate(uniq, 1):
        facts = pit.companyfacts(t)
        if not facts:
            continue
        for d in pit.filing_dates(t, start, end, facts=facts):
            rows.append({"ticker": t, "event_date": d, "anchor": "filing"})
        if verbose and i % 50 == 0:
            print(f"    filings {i}/{len(uniq)}", flush=True)
    return pd.DataFrame(rows)
