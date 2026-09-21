"""ClinicalTrials.gov API v2. Joined to BPC rows on the NCT number."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ..http import get

API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ("NCTId,OverallStatus,Phase,EnrollmentCount,DesignAllocation,"
          "DesignMasking,PrimaryCompletionDate,StudyType,LeadSponsorName")
BATCH = 50


def _flatten(study: dict) -> dict:
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    design = ps.get("designModule", {})
    sponsor = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    info = design.get("designInfo", {})
    return {
        "nct_id": ident.get("nctId"),
        "overall_status": status.get("overallStatus"),
        "phases": ",".join(design.get("phases", []) or []),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "allocation": info.get("allocation"),
        "masking": (info.get("maskingInfo") or {}).get("masking"),
        "primary_completion": (status.get("primaryCompletionDateStruct") or {}).get("date"),
        "study_type": design.get("studyType"),
        "lead_sponsor": sponsor.get("name"),
    }


def fetch(nct_ids: list[str]) -> pd.DataFrame:
    """Fetch trial design facts for a list of NCT ids, batched."""
    ids = sorted({n.strip() for n in nct_ids
                  if n and str(n).strip().upper().startswith("NCT")})
    if not ids:
        return pd.DataFrame()

    out = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        payload = get(API, params={
            "filter.ids": ",".join(chunk),
            "fields": FIELDS,
            "pageSize": BATCH,
        }).json()
        out.extend(_flatten(s) for s in payload.get("studies", []))

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").astype("Int64")
    df["pulled_at"] = dt.datetime.now()
    return df.drop_duplicates(subset=["nct_id"])
