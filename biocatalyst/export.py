"""Export the board to static JSON for hosting.

The board is a few hundred rows that take under a tenth of a second to score,
and it only changes when the pipeline runs. Serving that through a Python
function would mean a ~136MB bundle and a cold start per visitor to render
data that is identical all day, so the site is static and the filtering is
done in the browser.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from . import features, outcomes as outcomes_mod, score
from .config import ROOT
from .db import connect

WEB_DIR = ROOT / "web"
ROBINHOOD = "https://robinhood.com/stocks/{}"


def _clean(v):
    """JSON-safe scalar: NaN and NaT become null."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return v.date().isoformat()          # drop the 00:00:00 suffix
    if isinstance(v, dt.date):
        return v.isoformat()
    if pd.api.types.is_scalar(v) and pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def build_payload(horizon: int = 180, lookback: int = 45) -> dict:
    with connect(read_only=True) as con:
        feat = features.build(con, horizon_days=horizon)
    if feat.empty:
        return {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                "horizon": horizon, "rows": [],
                "recent": build_recent(lookback)}

    board = score.score(feat).sort_values(
        ["days_to_catalyst", "conviction"], ascending=[True, False])

    rows = []
    for _, r in board.iterrows():
        rows.append({
            "ticker": r["ticker"],
            "robinhood": ROBINHOOD.format(r["ticker"]),
            "company": _clean(r.get("company_name")),
            "drug": _clean(r.get("drug_name")),
            "indication": _clean(r.get("indication")),
            "stage": _clean(r.get("stage")),
            "date": _clean(r.get("catalyst_date_raw")),
            "days": _clean(r.get("days_to_catalyst")),
            "precision": _clean(r.get("date_precision")),
            "verdict": r["verdict"],
            "directional": bool(r["directional"]),
            "setup": r["setup"],
            "conviction": int(r["conviction"]),
            "evidence": r["evidence"],
            "evidence_note": r["evidence_note"],
            "loa": _clean(r.get("loa")),
            "implied": _clean(r.get("implied_move")),
            "expected": _clean(r.get("expected_move")),
            "runway": _clean(r.get("runway_months")),
            "ret20": _clean(r.get("ret_20d")),
            "mcap": _clean(r.get("market_cap")),
            "sentiment": _clean(r.get("sentiment")),
            "sent_n": _clean(r.get("scored_articles")),
            "sent_thin": bool(r.get("sentiment_thin")) if pd.notna(
                r.get("sentiment_thin")) else True,
            "eightk": _clean(r.get("eightk_90d")),
            "headline": _clean(r.get("top_negative")) or _clean(r.get("top_positive")),
            "reasons": r["reasons"],
            "nct": _clean(r.get("nct_id")),
        })

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "horizon": horizon,
        "count": len(rows),
        "rows": rows,
        "recent": build_recent(lookback),
    }


def build_recent(lookback: int = 45) -> list[dict]:
    """Catalysts that have already happened, with what the stock did."""
    with connect(read_only=True) as con:
        try:
            df = outcomes_mod.recent(con, lookback_days=lookback)
        except Exception:
            return []
    if df.empty:
        return []
    df = df.sort_values("catalyst_date", ascending=False)
    out = []
    for _, r in df.iterrows():
        out.append({
            "ticker": r["ticker"],
            "robinhood": ROBINHOOD.format(r["ticker"]),
            "drug": _clean(r.get("drug_name")),
            "indication": _clean(r.get("indication")),
            "stage": _clean(r.get("stage")),
            "date": str(_clean(r.get("catalyst_date"))),
            "move_1d": _clean(r.get("move_1d")),
            "abn_1d": _clean(r.get("abn_1d")),
            "move_todate": _clean(r.get("move_todate")),
            "abn_todate": _clean(r.get("abn_todate")),
            "last_date": str(_clean(r.get("last_date")) or ""),
            "prior_verdict": _clean(r.get("prior_verdict")),
            "prior_conviction": _clean(r.get("prior_conviction")),
        })
    return out


def export(out_dir: Path | None = None, horizon: int = 180,
           lookback: int = 45) -> Path:
    out = Path(out_dir or WEB_DIR)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(horizon, lookback)
    path = out / "board.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
