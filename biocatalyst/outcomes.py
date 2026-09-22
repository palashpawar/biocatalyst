"""What actually happened after a catalyst that has already passed.

Deliberately reuses the backtest's price panel and benchmark, so a move shown
on the board is measured the same way as a move in the validated studies:
abnormal against XBI, because a biotech that fell 8% on a day the sector fell
9% did not sell off on its news.

The verdict attached to an outcome is the last one recorded *before* the
catalyst date. Re-deriving it today would let the engine mark its own homework
with the answer in hand.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .backtest.engine import BENCHMARK, build_price_panel

# A catalyst stated only as "Q4 2026" has no usable event date; scoring the
# move around an arbitrary day inside that range would be noise.
USABLE_PRECISION = {"day", "month_part"}


def _pos_on_or_before(index: pd.DatetimeIndex, day: dt.date) -> int | None:
    loc = index.searchsorted(pd.Timestamp(day), side="right") - 1
    return int(loc) if loc >= 0 else None


def compute(con, lookback_days: int = 45, verbose: bool = False) -> pd.DataFrame:
    """Realised moves for catalysts whose date has passed inside the window."""
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)

    past = con.execute("""
        SELECT drug_id, ticker, catalyst_date_lo, date_precision
        FROM catalysts
        WHERE catalyst_date_hi < current_date
          AND catalyst_date_lo >= ?
          AND ticker IS NOT NULL
    """, [start]).fetchdf()
    if past.empty:
        return pd.DataFrame()

    past["catalyst_date_lo"] = pd.to_datetime(past["catalyst_date_lo"]).dt.date
    past = past[past["date_precision"].isin(USABLE_PRECISION)]
    if past.empty:
        return pd.DataFrame()

    tickers = sorted(past["ticker"].dropna().unique())
    if verbose:
        print(f"    outcomes: {len(past)} passed catalysts, "
              f"{len(tickers)} tickers", flush=True)

    closes, _, _ = build_price_panel(
        tickers, start - dt.timedelta(days=30), today, verbose=False)
    if closes.empty or BENCHMARK not in closes.columns:
        return pd.DataFrame()

    idx = closes.index
    bench = closes[BENCHMARK]
    rows = []

    for _, r in past.iterrows():
        t = r["ticker"]
        if t not in closes.columns:
            continue
        px = closes[t].dropna()
        if px.empty:
            continue

        # Enter on the last close before the catalyst: the move is what the
        # news did, not what the day it was announced had already done.
        entry = _pos_on_or_before(idx, r["catalyst_date_lo"] - dt.timedelta(days=1))
        if entry is None:
            continue
        p0, b0 = closes[t].iloc[entry], bench.iloc[entry]
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(b0) and b0 > 0):
            continue

        rec = {
            "drug_id": int(r["drug_id"]),
            "catalyst_date": r["catalyst_date_lo"],
            "ticker": t,
            "entry_date": idx[entry].date(),
            "entry_price": float(p0),
            "computed_at": dt.datetime.now(),
        }

        for label, offset in (("1d", 1), ("5d", 5)):
            j = entry + offset
            if j < len(idx) and np.isfinite(closes[t].iloc[j]):
                p1, b1 = closes[t].iloc[j], bench.iloc[j]
                rec[f"move_{label}"] = float(p1 / p0 - 1)
                rec[f"abn_{label}"] = float((p1 / p0 - 1) - (b1 / b0 - 1))
            else:
                rec[f"move_{label}"] = rec[f"abn_{label}"] = np.nan

        last = int(np.where(np.isfinite(closes[t].to_numpy()))[0].max())
        if last > entry:
            p1, b1 = closes[t].iloc[last], bench.iloc[last]
            rec["last_price"] = float(p1)
            rec["last_date"] = idx[last].date()
            rec["move_todate"] = float(p1 / p0 - 1)
            rec["abn_todate"] = float((p1 / p0 - 1) - (b1 / b0 - 1))
        else:
            rec["last_price"] = rec["move_todate"] = rec["abn_todate"] = np.nan
            rec["last_date"] = None

        rows.append(rec)

    return pd.DataFrame(rows)


def snapshot_verdicts(con, board: pd.DataFrame) -> pd.DataFrame:
    """Record today's verdicts so outcomes can be scored honestly later."""
    if board.empty:
        return pd.DataFrame()
    today = dt.date.today()
    out = pd.DataFrame({
        "drug_id": board["drug_id"].astype("int64"),
        "snapshot_date": today,
        "ticker": board["ticker"],
        "catalyst_date": board["catalyst_date_lo"],
        "verdict": board["verdict"],
        "setup": board["setup"],
        "conviction": board["conviction"].astype("int64"),
        "evidence": board["evidence"],
        "pulled_at": dt.datetime.now(),
    })
    return out.drop_duplicates(subset=["drug_id", "snapshot_date"])


def recent(con, lookback_days: int = 45) -> pd.DataFrame:
    """Passed catalysts joined to the verdict that was live beforehand."""
    return con.execute("""
        SELECT o.*, c.drug_name, c.indication, c.stage, c.company_name,
               c.catalyst_date_raw,
               v.verdict AS prior_verdict, v.setup AS prior_setup,
               v.conviction AS prior_conviction, v.snapshot_date AS verdict_asof
        FROM outcomes o
        LEFT JOIN catalysts c ON o.drug_id = c.drug_id
        LEFT JOIN (
            SELECT * FROM verdict_log
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY drug_id
                ORDER BY snapshot_date DESC
            ) = 1
        ) v ON o.drug_id = v.drug_id AND v.snapshot_date < o.catalyst_date
        WHERE o.catalyst_date >= current_date - INTERVAL (?) DAY
        ORDER BY o.catalyst_date DESC
    """, [lookback_days]).fetchdf()
