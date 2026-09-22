"""Build a catalyst calendar from primary sources, with no paywall in the way.

BiopharmCatalyst's free tier exposes only the first ten rows of its calendar;
every other row arrives with `blur: true` and its catalyst fields nulled out.
Rather than work around that, this module reconstructs the same calendar from
the sources BPC itself aggregates:

  * PDUFA / regulatory dates  -> EDGAR full-text search over 8-K filings
  * Clinical readouts         -> ClinicalTrials.gov primary completion dates

The result is less curated than BPC's -- no analyst has cleaned it -- but it is
public, complete, and free to use.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
from functools import lru_cache

import pandas as pd

from ..baserates import as_text
from ..http import get
from .bpc import parse_catalyst_date

EFTS = "https://efts.sec.gov/LATEST/search-index"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"

_SUFFIXES = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|limited|plc|holdings|holding|"
    r"group|sa|nv|ag|ab|as|oyj|therapeutics|pharmaceuticals|pharmaceutical|"
    r"pharma|biosciences|bioscience|biopharmaceuticals|biopharma|labs|"
    r"laboratories|technologies|sciences)\b", re.I)

_PDUFA_RE = re.compile(
    r"(?:PDUFA|target action|action)\s+(?:goal\s+)?date[^.]{0,60}?"
    r"((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4})", re.I)


def _norm(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    s = _SUFFIXES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _sec_name_index() -> dict[str, str]:
    """Normalised company name -> ticker, from SEC's official mapping."""
    data = get("https://www.sec.gov/files/company_tickers.json").json()
    idx: dict[str, str] = {}
    for row in data.values():
        key = _norm(row["title"])
        if key and key not in idx:
            idx[key] = row["ticker"].upper()
    return idx


def sponsor_to_ticker(sponsor: str | None) -> str | None:
    """Best-effort map from a trial sponsor name to a listed ticker."""
    if not sponsor:
        return None
    idx = _sec_name_index()
    key = _norm(sponsor)
    if not key:
        return None
    if key in idx:
        return idx[key]
    # Fall back to the longest indexed name that is a token-prefix match.
    best, best_len = None, 0
    for name, tick in idx.items():
        if (key.startswith(name + " ") or name.startswith(key + " ")) and len(name) > best_len:
            best, best_len = tick, len(name)
    return best


def pdufa_calendar(months_back: int = 12, limit: int = 300) -> pd.DataFrame:
    """Find announced PDUFA dates by full-text searching recent 8-K filings."""
    end = dt.date.today()
    start = end - dt.timedelta(days=30 * months_back)
    rows, offset = [], 0

    while offset < limit:
        payload = get(EFTS, params={
            "q": '"PDUFA date"', "forms": "8-K",
            "startdt": start.isoformat(), "enddt": end.isoformat(),
            "from": offset,
        }).json()
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h.get("_source", {})
            names = src.get("display_names") or []
            ticker = None
            if names:
                m = re.search(r"\(([A-Z][A-Z0-9.\-]{0,5})(?:,|\))", names[0])
                if m:
                    ticker = m.group(1)
            rows.append({
                "ticker": ticker,
                "company_name": names[0].split("  (")[0] if names else None,
                "filed": src.get("file_date"),
                "cik": (src.get("ciks") or [None])[0],
                "doc_id": h.get("_id"),
            })
        offset += len(hits)
        if len(hits) < 10:
            break

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["ticker", "filed"]) if not df.empty else df


_DRUG_PATTERNS = [
    # "NDA for XYZ", "BLA for INO-3107", "application for Pitolisant GR"
    re.compile(r"\b(?:s?NDA|s?BLA|NDA|BLA|application|submission)\s+for\s+"
               r"([A-Z][A-Za-z0-9\-]{2,28}(?:\s+[A-Z]{1,3}\b)?)"),
    # "Pitolisant GR NDA", "AXPAXLI NDA"
    re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,28}(?:\s+[A-Z]{1,3}\b)?)\s+"
               r"(?:s?NDA|s?BLA)\b"),
    # Development codes: INO-3107, BP-205, SRP-9001
    re.compile(r"\b([A-Z]{2,6}-\d{2,5})\b"),
]

# EDGAR document furniture that looks like a development code. "EX-99.1" sits
# at the top of every exhibit and was being read as the drug.
_NOT_A_DRUG = re.compile(
    r"^(?:EX|EXHIBIT|FORM|ITEM|SEC|CIK|SIC|IRS|CUSIP|ISIN|NYSE|NASDAQ|US|GAAP|"
    r"Q\d|FY|ASC|ASU|IPO|CEO|CFO|COO|FDA|EMA|CHMP|NDA|BLA|IND|PDUFA)"
    r"(?:[-\d].*)?$", re.I)

_INDICATION_PATTERNS = [
    re.compile(r"(?:for the treatment of|as a treatment for|as treatment for|"
               r"for treatment of|indicated for the treatment of|indicated for|"
               r"in patients with|in adults with|for patients with)\s+"
               r"([^.;:\u2022|()]{4,140})", re.I),
]

# Boilerplate that shows up where a drug name should be.
_DRUG_STOPWORDS = {
    "the", "this", "our", "its", "a", "an", "new", "drug", "marketing",
    "supplemental", "biologics", "license", "priority", "review", "company",
    "fda", "us", "u.s.", "approval", "acceptance", "filing", "submission",
    "product", "candidate", "both", "these", "their",
}

# How far from the PDUFA mention an indication may sit. Press releases cover
# several programs, so scanning the whole document reliably picks up the wrong
# drug's disease. A missing indication costs only the flat base rate; a wrong
# one produces a confidently wrong prior, which is worse.
_INDICATION_WINDOW = 900

# Phrases that mean the match ran into a quote, an attribution or a new topic.
_INDICATION_CUT = re.compile(
    r"[\u201c\u201d\"]|\bsaid\b|\bChairman\b|\bChief\b|\bOfficer\b|"
    r"\bPresident\b|\bCash\b|\bset a\b|\baccepted by\b|\bstarting with\b|"
    r"\bfollowed by\b|\bin Q[1-4]\b|\bapproximately\b")

# Too vague to classify, or a molecular target rather than a disease.
_INDICATION_REJECT = re.compile(
    r"^(?:symptomatic|patients?|adults?|treatment|the disease|disease|"
    r"this indication|such patients)$|growth factor receptor|"
    r"^(?:moderate|severe|mild)(?:\s+to\s+\w+)?$", re.I)

_INDICATION_LEAD = re.compile(
    r"^(?:adults?|adult|patients?|people|individuals|children|pediatric"
    r"|subjects?)\s+(?:with|who|aged)\s+", re.I)


def _clean_filing(raw: str) -> str:
    txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt)


def _tidy_indication(text: str) -> str | None:
    """Trim a captured phrase down to something classify_ta can use."""
    out = text.strip(" ,;-\u2013\u2014")
    out = _INDICATION_LEAD.sub("", out)
    # Cut at the first connective that starts a new clause.
    out = re.split(r"\s+(?:and is|which|that|who|based on|following|after|"
                   r"in the|under|with a|pursuant|announced|reported|"
                   r"is expected|has been|was granted)\b",
                   out, maxsplit=1)[0]
    # Trailing subordinate clauses: ", and assigned a PDUFA ...",
    # ", to provide an effective ...", ", including ... , to ..."
    out = re.split(r",\s+(?:and|to|which|with|as|for|has|is|was|assigned|"
                   r"providing|provided)\b", out, maxsplit=1)[0]
    out = re.sub(r"\s+(?:in|for)\s+(?:adults?|children|patients?|"
                 r"pediatric(?:s)?)\s*$", "", out, flags=re.I)
    cut = _INDICATION_CUT.search(out)
    if cut:
        out = out[:cut.start()]
    out = out.strip(" ,;-\u2013\u2014")
    # Real indications are short. Anything longer has run into narrative.
    words = out.split()
    if len(words) > 10:
        out = " ".join(words[:10])
    # Cutting mid-clause can leave a dangling connective ("presbyopia and").
    out = re.sub(r"\s+(?:and|or|with|for|in|of|to|the|a|an|due|associated)$",
                 "", out.strip(" ,;-"), flags=re.I).strip(" ,;-")
    if len(out) < 4 or len(out.split()) < 1:
        return None
    if not re.search(r"[A-Za-z]{4}", out):
        return None
    if _INDICATION_REJECT.search(out):
        return None
    return out


def _nearest(matches, anchor: int):
    """The match whose position sits closest to the PDUFA mention."""
    best, best_dist = None, None
    for m in matches:
        dist = abs(m.start() - anchor)
        if best_dist is None or dist < best_dist:
            best, best_dist = m, dist
    return best


def extract_pdufa_details(cik: str, doc_id: str) -> dict | None:
    """Pull the PDUFA date, drug and indication out of a filing in one fetch.

    The indication is the point: without it every PDUFA row falls back to the
    flat 90% base rate with no therapeutic-area adjustment, so an oncology
    filing and a rare-disease filing score identically.
    """
    try:
        adsh, _, fname = (doc_id or "").partition(":")
        if not adsh or not fname:
            return None
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{adsh.replace('-', '')}/{fname}")
        text = _clean_filing(get(url).text)
    except Exception:
        return None

    dm = _PDUFA_RE.search(text)
    if not dm:
        return None
    lo, _, _ = parse_catalyst_date(dm.group(1).replace(",", ""))
    if not lo:
        return None
    anchor = dm.start()

    # Look for the drug near the PDUFA mention before falling back to the doc.
    window = text[max(0, anchor - 1200):anchor + 600]
    drug = None
    for pat in _DRUG_PATTERNS:
        for m in pat.finditer(window):
            cand = m.group(1).strip()
            if cand.lower() in _DRUG_STOPWORDS:
                continue
            if cand.lower().split()[0] in _DRUG_STOPWORDS:
                continue
            if _NOT_A_DRUG.match(cand):
                continue
            drug = cand
            break
        if drug:
            break

    # Only trust matches close to the PDUFA mention.
    lo_w = max(0, anchor - _INDICATION_WINDOW)
    hi_w = anchor + _INDICATION_WINDOW
    indication = None
    for pat in _INDICATION_PATTERNS:
        hits = [m for m in pat.finditer(text) if lo_w <= m.start() <= hi_w]
        while hits:
            m = _nearest(hits, anchor)
            indication = _tidy_indication(m.group(1))
            if indication:
                break
            hits.remove(m)
        if indication:
            break

    return {"catalyst_date": lo, "drug_name": drug, "indication": indication}


def extract_pdufa_date(cik: str, doc_id: str) -> dt.date | None:
    """Back-compatible wrapper returning only the date."""
    d = extract_pdufa_details(cik, doc_id)
    return d["catalyst_date"] if d else None


def _simplify_stage(raw: str | None) -> str:
    """Collapse ClinicalTrials' phase list into one base-rate bucket.

    A combined PHASE1,PHASE2 trial is priced off the *earlier* phase: treating
    it as a clean Phase 2 would hand it Phase 2's much higher approval prior.
    """
    s = as_text(raw).lower().replace(" ", "")
    has = lambda p: p in s
    if has("phase2") and has("phase3"):
        return "phase2/3"
    if has("phase1") and has("phase2"):
        return "phase1/2"
    for p in ("phase4", "phase3", "phase2", "phase1"):
        if has(p):
            return p
    return "other"


def trial_readouts(horizon_days: int = 180, phases=("PHASE2", "PHASE3"),
                   max_studies: int = 1000,
                   lookback_days: int = 0) -> pd.DataFrame:
    """Industry-sponsored trials whose primary completion falls in the window.

    `lookback_days` reaches back past today so recently-passed catalysts can be
    scored against what the stock actually did.
    """
    today = dt.date.today()
    start_window = today - dt.timedelta(days=lookback_days)
    end = today + dt.timedelta(days=horizon_days)
    rows, token = [], None

    while len(rows) < max_studies:
        params = {
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
            "query.term": "AREA[LeadSponsorClass]INDUSTRY",
            "filter.advanced": (
                f"AREA[PrimaryCompletionDate]RANGE[{start_window},{end}]"),
            "fields": ("NCTId,BriefTitle,Phase,PrimaryCompletionDate,"
                       "LeadSponsorName,EnrollmentCount,Condition,OverallStatus"),
            "pageSize": 200,
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
            rows.append({
                "nct_id": ps["identificationModule"]["nctId"],
                "drug_name": ps["identificationModule"].get("briefTitle"),
                "stage": ",".join(ph),
                "primary_completion": (ps["statusModule"]
                                       .get("primaryCompletionDateStruct", {})
                                       .get("date")),
                "lead_sponsor": ps["sponsorCollaboratorsModule"]["leadSponsor"]["name"],
                "indication": "; ".join(conds[:3]),
                "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
            })
        token = payload.get("nextPageToken")
        if not token:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ticker"] = df["lead_sponsor"].map(sponsor_to_ticker)
    return df


def build_calendar(horizon_days: int = 180, with_pdufa: bool = True,
                   verbose: bool = False,
                   lookback_days: int = 45) -> pd.DataFrame:
    """Combined catalyst calendar in the same shape as the BPC table."""
    frames = []

    if verbose:
        print("    ClinicalTrials.gov readouts ...", flush=True)
    tr = trial_readouts(horizon_days=horizon_days,
                        lookback_days=lookback_days)
    if not tr.empty:
        tr = tr[tr["ticker"].notna()].copy()
        lohi = tr["primary_completion"].map(parse_catalyst_date)
        tr["catalyst_date_lo"] = [x[0] for x in lohi]
        tr["catalyst_date_hi"] = [x[1] for x in lohi]
        tr["date_precision"] = [x[2] for x in lohi]
        tr["catalyst_date_raw"] = tr["primary_completion"]
        tr["simplified_stage"] = tr["stage"].map(_simplify_stage)
        tr["next_catalyst"] = "Primary completion"
        tr["company_name"] = tr["lead_sponsor"]
        tr["source"] = "clinicaltrials"
        frames.append(tr)
        if verbose:
            print(f"      {len(tr)} readouts mapped to tickers")

    if with_pdufa:
        if verbose:
            print("    EDGAR PDUFA announcements ...", flush=True)
        pd_df = pdufa_calendar()
        if not pd_df.empty:
            pd_df = pd_df[pd_df["ticker"].notna()].copy()
            details = [extract_pdufa_details(r["cik"], r["doc_id"])
                       for _, r in pd_df.iterrows()]
            pd_df["catalyst_date_lo"] = [d["catalyst_date"] if d else None
                                         for d in details]
            pd_df["catalyst_date_hi"] = pd_df["catalyst_date_lo"]
            pd_df["drug_name"] = [(d or {}).get("drug_name") or "(see 8-K)"
                                  for d in details]
            pd_df["indication"] = [(d or {}).get("indication") for d in details]
            pd_df = pd_df[pd_df["catalyst_date_lo"].notna()].copy()
            pd_df["date_precision"] = "day"
            pd_df["catalyst_date_raw"] = pd_df["catalyst_date_lo"].astype(str)
            pd_df["simplified_stage"] = "pdufa"
            pd_df["stage"] = "PDUFA"
            pd_df["next_catalyst"] = "PDUFA date"
            pd_df["nct_id"] = None
            pd_df["source"] = "edgar"
            frames.append(pd_df)
            if verbose:
                print(f"      {len(pd_df)} PDUFA dates extracted")

    if not frames:
        return pd.DataFrame()

    cal = pd.concat(frames, ignore_index=True, sort=False)
    today = dt.date.today()
    floor = today - dt.timedelta(days=lookback_days)
    cal = cal[cal["catalyst_date_hi"].map(lambda d: pd.notna(d) and d >= floor)]
    # A stable, content-derived id. A row counter would hand the same catalyst
    # a different id on every refresh, so the (drug_id, catalyst_date_raw)
    # primary key would never match and each run would duplicate the table.
    def _stable_id(r) -> int:
        # Identity only: ticker, trial and date. Hashing drug_name as well
        # meant that improving the extractor changed the id, so the row was
        # inserted alongside its old version instead of replacing it and the
        # table accumulated stale duplicates on every refresh.
        key = f"{r.get('ticker')}|{r.get('nct_id')}|{r.get('catalyst_date_raw')}"
        return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)

    cal["drug_id"] = cal.apply(_stable_id, axis=1)
    cal["designations"] = ""
    cal["fda_status"] = cal.get("OverallStatus")
    cal["optionable"] = None  # unknown; resolved when pricing chains
    cal["price"] = None
    cal["note"] = cal["source"]
    cal["pulled_at"] = dt.datetime.now()

    keep = ["drug_id", "ticker", "company_name", "drug_name", "nct_id",
            "indication", "stage", "simplified_stage", "fda_status",
            "next_catalyst", "catalyst_date_raw", "catalyst_date_lo",
            "catalyst_date_hi", "date_precision", "designations", "note",
            "optionable", "price", "pulled_at"]
    for c in keep:
        if c not in cal.columns:
            cal[c] = None
    cal = cal[keep]

    # Several 8-Ks often announce the same PDUFA date. Once drug names started
    # being extracted they stopped being identical, so those rows no longer
    # collapsed. Keep one row per ticker and date -- the most informative.
    cal = cal.assign(
        _rank=cal["indication"].notna().astype(int) * 2
        + (cal["drug_name"] != "(see 8-K)").astype(int)
    ).sort_values("_rank", ascending=False)
    cal = cal.drop_duplicates(subset=["ticker", "catalyst_date_raw", "nct_id"])
    pdufa = cal["simplified_stage"] == "pdufa"
    cal = pd.concat([
        cal[~pdufa],
        cal[pdufa].drop_duplicates(subset=["ticker", "catalyst_date_raw"]),
    ]).drop(columns=["_rank"])
    return cal
