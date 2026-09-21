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

from . import features, score
from .config import ROOT
from .db import connect

WEB_DIR = ROOT / "web"
ROBINHOOD = "https://robinhood.com/stocks/{}"


def _clean(v):
    """JSON-safe scalar: NaN and NaT become null."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, dt.date, dt.datetime)):
        return str(v)
    if pd.api.types.is_scalar(v) and pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def build_payload(horizon: int = 180) -> dict:
    with connect(read_only=True) as con:
        feat = features.build(con, horizon_days=horizon)
    if feat.empty:
        return {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                "horizon": horizon, "rows": []}

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
    }


def export(out_dir: Path | None = None, horizon: int = 180) -> Path:
    out = Path(out_dir or WEB_DIR)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(horizon)
    path = out / "board.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
