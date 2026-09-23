import datetime as dt
import sys, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst import outcomes
from biocatalyst.backtest.engine import BENCHMARK


def test_only_precise_dates_are_scored():
    # A catalyst stated as "Q4 2026" has no event day; measuring a move around
    # an arbitrary date inside that range would be noise.
    assert "day" in outcomes.USABLE_PRECISION
    assert "month_part" in outcomes.USABLE_PRECISION
    for vague in ("quarter", "half", "year", "month", "unknown"):
        assert vague not in outcomes.USABLE_PRECISION


def test_entry_is_the_last_close_before_the_catalyst():
    idx = pd.bdate_range("2026-09-01", periods=20)
    # Catalyst on a Wednesday: entry must be the prior trading day.
    target = dt.date(2026, 9, 16)
    pos = outcomes._pos_on_or_before(idx, target - dt.timedelta(days=1))
    assert idx[pos].date() < target
    assert idx[pos + 1].date() >= target


def test_entry_handles_a_date_before_all_history():
    idx = pd.bdate_range("2026-09-01", periods=5)
    assert outcomes._pos_on_or_before(idx, dt.date(2020, 1, 1)) is None


def test_entry_falls_back_over_a_weekend():
    idx = pd.bdate_range("2026-09-01", periods=20)
    # Monday catalyst -> entry is the preceding Friday, not a missing Sunday.
    monday = dt.date(2026, 9, 21)
    pos = outcomes._pos_on_or_before(idx, monday - dt.timedelta(days=1))
    assert idx[pos].weekday() == 4


def _con_with(rows):
    import duckdb
    from biocatalyst.db import SCHEMA
    con = duckdb.connect(":memory:")
    con.execute(SCHEMA)
    if rows:
        con.executemany(
            "INSERT INTO catalysts (drug_id, ticker, catalyst_date_lo, "
            "catalyst_date_hi, date_precision, catalyst_date_raw) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(*r, str(r[2])) for r in rows])
    return con


def test_vague_dates_are_excluded_from_the_query():
    past = dt.date.today() - dt.timedelta(days=5)
    con = _con_with([
        (1, "AAA", past, past, "day"),
        (2, "BBB", past, past, "quarter"),
    ])
    # No network in tests: compute() returns empty once the panel is empty,
    # but the precision filter must have already dropped BBB.
    df = con.execute(
        "SELECT ticker FROM catalysts WHERE catalyst_date_hi < current_date"
    ).fetchdf()
    kept = [t for t, p in zip(df["ticker"], ["day", "quarter"])
            if p in outcomes.USABLE_PRECISION]
    assert kept == ["AAA"]
    con.close()


def test_no_past_catalysts_returns_empty():
    future = dt.date.today() + dt.timedelta(days=20)
    con = _con_with([(1, "AAA", future, future, "day")])
    assert outcomes.compute(con).empty
    con.close()


def test_snapshot_verdicts_shape():
    board = pd.DataFrame({
        "drug_id": [11, 22], "ticker": ["AAA", "BBB"],
        "catalyst_date_lo": [dt.date(2026, 10, 1), dt.date(2026, 10, 2)],
        "verdict": ["SHORT", "NO TRADE"], "setup": ["DILUTION_SHORT", "NO_EDGE"],
        "conviction": [60, 5], "evidence": ["supported", "n/a"]})
    out = outcomes.snapshot_verdicts(None, board)
    assert len(out) == 2
    assert out["snapshot_date"].iloc[0] == dt.date.today()
    assert set(out.columns) >= {"drug_id", "snapshot_date", "verdict", "conviction"}


def test_snapshot_of_empty_board():
    assert outcomes.snapshot_verdicts(None, pd.DataFrame()).empty


def test_benchmark_is_the_shared_one():
    # Outcomes must use the same benchmark as the backtest, or a move on the
    # board would not be comparable to a move in the validated studies.
    assert BENCHMARK == "XBI"


def test_site_grid_columns_match_cells_per_row():
    # The table became a grid, but the same failure mode exists: if
    # grid-template-columns and the number of cells in a row disagree, every
    # column after the mismatch shifts. Previously this shipped as 10 cells
    # under 9 headers.
    import re
    html = (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text()

    tmpl = re.search(r"\.row\{[^}]*grid-template-columns:([^;]+);", html)
    assert tmpl, "row grid template not found"
    n_cols = len(tmpl.group(1).split())

    body = html[html.index("function rowHtml"):html.index("function detailHtml")]
    n_cells = len(re.findall(r'<div class="cell-', body))
    assert n_cols == n_cells, f"{n_cols} grid columns vs {n_cells} cells"


def test_recent_grid_columns_match_cells():
    import re
    html = (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text()
    body = html[html.index("function renderRecent"):]
    inline = re.search(r'grid-template-columns:([^"]+)"', body)
    assert inline, "recent grid template not found"
    n_cols = len(inline.group(1).split())
    row = body[body.index('<div class="row"'):body.index("`).join")]
    n_cells = len(re.findall(r"<div(?: class=\"(?:conv)\")?>", row))
    assert n_cells == n_cols, f"{n_cols} columns vs {n_cells} cells"
