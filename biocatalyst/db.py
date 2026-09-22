"""DuckDB access. Single-writer: open one connection, close it before another opens."""
from __future__ import annotations

import contextlib
from typing import Iterator

import duckdb

from .config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalysts (
    drug_id                BIGINT,
    ticker                 VARCHAR,
    company_name           VARCHAR,
    drug_name              VARCHAR,
    nct_id                 VARCHAR,
    indication             VARCHAR,
    stage                  VARCHAR,
    simplified_stage       VARCHAR,
    fda_status             VARCHAR,
    next_catalyst          VARCHAR,
    catalyst_date_raw      VARCHAR,
    catalyst_date_lo       DATE,
    catalyst_date_hi       DATE,
    date_precision         VARCHAR,
    designations           VARCHAR,
    note                   VARCHAR,
    optionable             BOOLEAN,
    price                  DOUBLE,
    pulled_at              TIMESTAMP,
    PRIMARY KEY (drug_id, catalyst_date_raw)
);

CREATE TABLE IF NOT EXISTS trials (
    nct_id                 VARCHAR PRIMARY KEY,
    overall_status         VARCHAR,
    phases                 VARCHAR,
    enrollment             INTEGER,
    allocation             VARCHAR,
    masking                VARCHAR,
    primary_completion     VARCHAR,
    study_type             VARCHAR,
    lead_sponsor           VARCHAR,
    pulled_at              TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financials (
    ticker                 VARCHAR PRIMARY KEY,
    cik                    VARCHAR,
    cash_usd               DOUBLE,
    cash_asof              DATE,
    quarterly_burn_usd     DOUBLE,
    runway_months          DOUBLE,
    shares_outstanding     DOUBLE,
    pulled_at              TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices (
    ticker                 VARCHAR PRIMARY KEY,
    last                   DOUBLE,
    ret_20d                DOUBLE,
    ret_60d                DOUBLE,
    rvol_20d               DOUBLE,
    rvol_60d               DOUBLE,
    adv_usd                DOUBLE,
    pct_of_52w_high        DOUBLE,
    market_cap             DOUBLE,
    pulled_at              TIMESTAMP
);

-- What the board said about a catalyst *before* it happened. Scoring the
-- engine against outcomes it can already see would be worthless, so verdicts
-- are snapshotted on every refresh and the outcome joins the last snapshot
-- taken before the catalyst date.
CREATE TABLE IF NOT EXISTS verdict_log (
    drug_id                BIGINT,
    snapshot_date          DATE,
    ticker                 VARCHAR,
    catalyst_date          DATE,
    verdict                VARCHAR,
    setup                  VARCHAR,
    conviction             INTEGER,
    evidence               VARCHAR,
    pulled_at              TIMESTAMP,
    PRIMARY KEY (drug_id, snapshot_date)
);

-- Realised move around a catalyst that has already passed.
CREATE TABLE IF NOT EXISTS outcomes (
    drug_id                BIGINT,
    catalyst_date          DATE,
    ticker                 VARCHAR,
    entry_date             DATE,
    entry_price            DOUBLE,
    last_price             DOUBLE,
    last_date              DATE,
    move_1d                DOUBLE,
    move_5d                DOUBLE,
    move_todate            DOUBLE,
    abn_1d                 DOUBLE,
    abn_5d                 DOUBLE,
    abn_todate             DOUBLE,
    computed_at            TIMESTAMP,
    PRIMARY KEY (drug_id, catalyst_date)
);

-- Forward sentiment log: one row per ticker per day, never deleted. Free news
-- sources have no point-in-time archive, so the only way to make sentiment
-- testable is to start recording it and wait.
CREATE TABLE IF NOT EXISTS sentiment (
    ticker                 VARCHAR,
    snapshot_date          DATE,
    articles               INTEGER,
    scored_articles        INTEGER,
    pos_hits               INTEGER,
    neg_hits               INTEGER,
    sentiment              DOUBLE,
    thin                   BOOLEAN,
    top_positive           VARCHAR,
    top_negative           VARCHAR,
    eightk_30d             INTEGER,
    eightk_90d             INTEGER,
    eightk_365d            INTEGER,
    days_since_8k          INTEGER,
    pulled_at              TIMESTAMP,
    PRIMARY KEY (ticker, snapshot_date)
);

CREATE TABLE IF NOT EXISTS implied (
    ticker                 VARCHAR,
    catalyst_date          DATE,
    expiry                 DATE,
    atm_iv                 DOUBLE,
    implied_move           DOUBLE,
    straddle_pct           DOUBLE,
    pulled_at              TIMESTAMP,
    PRIMARY KEY (ticker, catalyst_date)
);
"""


@contextlib.contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection, creating the schema on first write."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    try:
        if not read_only:
            con.execute(SCHEMA)
        yield con
    finally:
        con.close()


def upsert(con: duckdb.DuckDBPyConnection, table: str, df) -> int:
    """Replace rows by primary key. Empty frames are a no-op."""
    if df is None or len(df) == 0:
        return 0
    con.register("_staging", df)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _staging")
    con.unregister("_staging")
    return len(df)
