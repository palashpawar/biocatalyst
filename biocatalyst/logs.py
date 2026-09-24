"""Forward logs that survive CI.

The nightly job runs on a fresh GitHub runner with an empty DuckDB: `data/` is
gitignored and only the EDGAR cache is restored. Anything that exists to
*accumulate* -- the verdict log that grades the engine on calls made in
advance, and the sentiment log that makes sentiment testable later -- was
being rebuilt from nothing every night, so it could never accumulate at all.

These tables now live as CSV files under `logs/`, committed alongside
`web/board.json`. Each refresh loads them into DuckDB, appends, and writes them
back.

Appends are change-only. A verdict that has not moved is not rewritten every
day; the row in effect on any date is simply the last one logged on or before
it. That keeps the files to a size worth committing while losing nothing a
point-in-time join needs.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .baserates import is_missing
from .config import ROOT
from .db import upsert

LOG_DIR = ROOT / "logs"

# table -> (file, key, date columns, timestamp columns, change-detection fn)
VERDICTS = "verdict_log"
SENTIMENT = "sentiment"

FILES = {
    VERDICTS: LOG_DIR / "verdicts.csv",
    SENTIMENT: LOG_DIR / "sentiment.csv",
}
KEYS = {VERDICTS: "drug_id", SENTIMENT: "ticker"}
DATE_COLS = {
    VERDICTS: ["snapshot_date", "catalyst_date"],
    SENTIMENT: ["snapshot_date"],
}
TS_COLS = {VERDICTS: ["pulled_at"], SENTIMENT: ["pulled_at"]}


def _day(v) -> str | None:
    """A date as YYYY-MM-DD, whatever type it arrived as.

    DuckDB hands a DATE column back as a pandas Timestamp, whose str() carries
    a " 00:00:00" suffix the in-memory `date` does not. Comparing raw strings
    made every logged row differ from its own reload, so the change-only log
    would have quietly rewritten everything every night.
    """
    if is_missing(v):
        return None
    try:
        return pd.Timestamp(v).date().isoformat()
    except (TypeError, ValueError):
        return str(v)[:10]


def _signature_verdict(r) -> tuple:
    # Conviction moves a point or two as a catalyst approaches; only a shift of
    # a full decile is a change worth recording.
    conv = r.get("conviction")
    bucket = None if is_missing(conv) else int(conv) // 10
    return (r.get("verdict"), r.get("setup"), bucket, r.get("evidence"),
            _day(r.get("catalyst_date")))


def _signature_sentiment(r) -> tuple:
    s = r.get("sentiment")
    return (None if is_missing(s) else round(float(s), 2),
            None if is_missing(r.get("scored_articles")) else int(r["scored_articles"]),
            None if is_missing(r.get("eightk_90d")) else int(r["eightk_90d"]),
            bool(r.get("thin")) if not is_missing(r.get("thin")) else None)


SIGNATURE = {VERDICTS: _signature_verdict, SENTIMENT: _signature_sentiment}


def _read(table: str) -> pd.DataFrame:
    path = FILES[table]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in DATE_COLS[table]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    for c in TS_COLS[table]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load(con) -> dict[str, int]:
    """Seed DuckDB from the committed logs. Idempotent."""
    counts = {}
    for table in FILES:
        df = _read(table)
        counts[table] = upsert(con, table, df) if not df.empty else 0
    return counts


def latest_signatures(con, table: str) -> dict:
    """Last logged signature per key, from what is already in DuckDB."""
    key = KEYS[table]
    try:
        cur = con.execute(f"""
            SELECT * FROM {table}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {key}
                                       ORDER BY snapshot_date DESC) = 1
        """).fetchdf()
    except Exception:
        return {}
    sig = SIGNATURE[table]
    return {row[key]: sig(row) for _, row in cur.iterrows()}


def changed_rows(new: pd.DataFrame, previous: dict, table: str) -> pd.DataFrame:
    """Rows whose signature differs from the last one logged for that key."""
    if new is None or new.empty:
        return pd.DataFrame()
    key, sig = KEYS[table], SIGNATURE[table]
    keep = [previous.get(r[key]) != sig(r) for _, r in new.iterrows()]
    return new[keep]


def append(con, table: str, new: pd.DataFrame) -> int:
    """Append only what changed; return how many rows were written."""
    delta = changed_rows(new, latest_signatures(con, table), table)
    return upsert(con, table, delta) if not delta.empty else 0


def dump(con) -> dict[str, int]:
    """Write the logs back to CSV, sorted so diffs stay readable."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table, path in FILES.items():
        df = con.execute(f"SELECT * FROM {table}").fetchdf()
        if df.empty:
            counts[table] = 0
            continue
        df = df.sort_values(["snapshot_date", KEYS[table]])
        df.to_csv(path, index=False)
        counts[table] = len(df)
    return counts


def compact(con, table: str) -> int:
    """Rewrite a table as change-only. Used once to seed from dense history."""
    df = con.execute(f"SELECT * FROM {table} ORDER BY snapshot_date").fetchdf()
    if df.empty:
        return 0
    key, sig = KEYS[table], SIGNATURE[table]
    seen: dict = {}
    keep = []
    for _, r in df.iterrows():
        s = sig(r)
        keep.append(seen.get(r[key]) != s)
        seen[r[key]] = s
    kept = df[keep]
    con.execute(f"DELETE FROM {table}")
    upsert(con, table, kept)
    return len(df) - len(kept)
