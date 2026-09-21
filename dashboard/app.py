"""Local Flask dashboard over the catalyst database.

    python3 -m dashboard.app    ->    http://127.0.0.1:5057
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biocatalyst import features, score  # noqa: E402
from biocatalyst.db import connect  # noqa: E402

app = Flask(__name__)

ROBINHOOD = "https://robinhood.com/stocks/{}"


def load_board(horizon: int, setup: str | None, min_conv: int,
               sort: str) -> pd.DataFrame:
    try:
        with connect(read_only=True) as con:
            feat = features.build(con, horizon_days=horizon)
    except Exception:
        return pd.DataFrame()
    if feat.empty:
        return feat

    board = score.score(feat)
    if setup and setup != "ALL":
        if setup == "ACTIONABLE":
            board = board[board["setup"] != "NO_EDGE"]
        else:
            board = board[board["setup"] == setup]
    if min_conv:
        board = board[board["conviction"] >= min_conv]

    if sort == "conviction":
        return board.sort_values(["conviction", "days_to_catalyst"],
                                 ascending=[False, True])
    # Default: soonest catalyst first, conviction breaking ties.
    return board.sort_values(["days_to_catalyst", "conviction"],
                             ascending=[True, False])


def _pct(v, digits=0):
    return f"{v:.{digits}%}" if pd.notna(v) else "—"


def _when(days) -> str:
    if pd.isna(days):
        return "—"
    d = int(days)
    if d <= 0:
        return "today"
    if d == 1:
        return "tomorrow"
    if d < 7:
        return f"{d} days"
    if d < 60:
        return f"{d // 7}w"
    return f"{d // 30}mo"


@app.route("/")
def index():
    horizon = request.args.get("horizon", 180, type=int)
    setup = request.args.get("setup", "ACTIONABLE")
    min_conv = request.args.get("min_conv", 0, type=int)
    sort = request.args.get("sort", "date")

    board = load_board(horizon, setup, min_conv, sort)

    rows, counts = [], {}
    if not board.empty:
        counts = board["verdict"].value_counts().to_dict()
        for _, r in board.head(250).iterrows():
            tick = r["ticker"]
            rows.append({
                "ticker": tick,
                "robinhood": ROBINHOOD.format(tick),
                "drug": (r.get("drug_name") or "")[:64],
                "indication": (r.get("indication") or "")[:56],
                "stage": r.get("stage") or "—",
                "date": r.get("catalyst_date_raw") or "—",
                "when": _when(r.get("days_to_catalyst")),
                "precision": r.get("date_precision"),
                "verdict": r["verdict"],
                "directional": r["directional"],
                "setup": r["setup"].replace("_", " ").title(),
                "conviction": r["conviction"],
                "evidence": r["evidence"],
                "evidence_note": r["evidence_note"],
                "loa": _pct(r.get("loa")),
                "implied": _pct(r.get("implied_move")),
                "expected": _pct(r.get("expected_move")),
                "runway": (f"{r['runway_months']:.0f}mo"
                           if pd.notna(r.get("runway_months")) else "—"),
                "ret20": _pct(r.get("ret_20d")),
                "mcap": (f"${r['market_cap'] / 1e6:,.0f}M"
                         if pd.notna(r.get("market_cap")) else "—"),
                "reasons": r["reasons"],
                # Context only -- sentiment never feeds a verdict.
                "sentiment": (None if pd.isna(r.get("sentiment"))
                              else float(r["sentiment"])),
                "sent_thin": bool(r.get("sentiment_thin")) if pd.notna(
                    r.get("sentiment_thin")) else True,
                "sent_n": (0 if pd.isna(r.get("scored_articles"))
                           else int(r["scored_articles"])),
                "headline": (r.get("top_negative") or r.get("top_positive")
                             or ""),
                "eightk": (None if pd.isna(r.get("eightk_90d"))
                           else int(r["eightk_90d"])),
            })

    return render_template(
        "board.html", rows=rows, counts=counts, horizon=horizon,
        setup=setup, min_conv=min_conv, sort=sort, total=len(board),
        generated=dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        setups=["ACTIONABLE", "ALL", "DILUTION_SHORT", "VOL_SELL_RICH",
                "VOL_BUY_CHEAP", "RUNUP_FADE", "BASE_RATE_LONG", "NO_EDGE"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=False)
