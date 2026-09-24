import datetime as dt
import sys, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import duckdb

from biocatalyst import logs, outcomes
from biocatalyst.db import SCHEMA, upsert

ROOT = pathlib.Path(__file__).resolve().parent.parent
D1, D2, D3 = dt.date(2026, 9, 21), dt.date(2026, 9, 22), dt.date(2026, 9, 23)


def _con():
    con = duckdb.connect(":memory:")
    con.execute(SCHEMA)
    return con


def _verdict(drug_id, day, verdict="SHORT", conviction=40, setup="DILUTION_SHORT",
             catalyst=dt.date(2026, 10, 30)):
    return {"drug_id": drug_id, "snapshot_date": day, "ticker": "X",
            "catalyst_date": catalyst, "verdict": verdict, "setup": setup,
            "conviction": conviction, "evidence": "supported",
            "pulled_at": pd.Timestamp("2026-09-21 12:00")}


def test_unchanged_verdict_is_not_rewritten():
    con = _con()
    assert logs.append(con, "verdict_log", pd.DataFrame([_verdict(1, D1)])) == 1
    assert logs.append(con, "verdict_log", pd.DataFrame([_verdict(1, D2)])) == 0
    assert con.execute("select count(*) from verdict_log").fetchone()[0] == 1


def test_changed_verdict_is_appended():
    con = _con()
    logs.append(con, "verdict_log", pd.DataFrame([_verdict(1, D1)]))
    n = logs.append(con, "verdict_log",
                    pd.DataFrame([_verdict(1, D2, verdict="NO TRADE", setup="NO_EDGE")]))
    assert n == 1


def test_new_catalyst_is_appended():
    con = _con()
    logs.append(con, "verdict_log", pd.DataFrame([_verdict(1, D1)]))
    assert logs.append(con, "verdict_log",
                       pd.DataFrame([_verdict(1, D2), _verdict(2, D2)])) == 1


def test_small_conviction_drift_is_not_a_change():
    # Conviction ticks a point or two as the date nears; only a decile move
    # is recorded, or the log would rewrite every row every day.
    con = _con()
    logs.append(con, "verdict_log", pd.DataFrame([_verdict(1, D1, conviction=41)]))
    assert logs.append(con, "verdict_log",
                       pd.DataFrame([_verdict(1, D2, conviction=44)])) == 0
    assert logs.append(con, "verdict_log",
                       pd.DataFrame([_verdict(1, D3, conviction=53)])) == 1


def test_sentiment_change_detection_is_null_safe():
    con = _con()
    row = {"ticker": "X", "snapshot_date": D1, "sentiment": float("nan"),
           "scored_articles": pd.NA, "eightk_90d": None, "thin": True,
           "pulled_at": pd.Timestamp("2026-09-21")}
    assert logs.append(con, "sentiment", pd.DataFrame([row])) == 1
    # identical nulls on the next day must not count as a change
    assert logs.append(con, "sentiment",
                       pd.DataFrame([{**row, "snapshot_date": D2}])) == 0


def test_dump_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logs, "FILES", {
        "verdict_log": tmp_path / "verdicts.csv",
        "sentiment": tmp_path / "sentiment.csv"})

    con = _con()
    logs.append(con, "verdict_log",
                pd.DataFrame([_verdict(1, D1), _verdict(2, D1, verdict="LONG")]))
    written = logs.dump(con)
    assert written["verdict_log"] == 2

    fresh = _con()                       # what a CI runner starts with
    assert logs.load(fresh)["verdict_log"] == 2
    got = fresh.execute("select * from verdict_log order by drug_id").fetchdf()
    assert list(got["verdict"]) == ["SHORT", "LONG"]
    assert pd.Timestamp(got["snapshot_date"].iloc[0]).date() == D1

    # and appends after a load still dedupe against the loaded history
    assert logs.append(fresh, "verdict_log", pd.DataFrame([_verdict(1, D2)])) == 0


def test_load_with_no_files_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, "FILES", {
        "verdict_log": tmp_path / "none1.csv", "sentiment": tmp_path / "none2.csv"})
    assert logs.load(_con()) == {"verdict_log": 0, "sentiment": 0}


def test_compact_turns_daily_history_into_changes():
    con = _con()
    dense = pd.DataFrame([_verdict(1, D1), _verdict(1, D2), _verdict(1, D3,
                          verdict="NO TRADE", setup="NO_EDGE")])
    upsert(con, "verdict_log", dense)
    assert logs.compact(con, "verdict_log") == 1        # D2 was a repeat
    days = [r[0] for r in con.execute(
        "select snapshot_date from verdict_log order by 1").fetchall()]
    assert days == [D1, D3]


def test_recent_uses_last_snapshot_before_the_catalyst():
    # Regression: the join took the latest snapshot per catalyst and THEN
    # required it to predate the event. A PDUFA is still on the board the
    # morning it happens, so its latest snapshot is the event day itself and
    # every such row lost its verdict.
    con = _con()
    event = dt.date.today() - dt.timedelta(days=5)
    upsert(con, "verdict_log", pd.DataFrame([
        _verdict(7, event - dt.timedelta(days=3), verdict="SHORT"),
        _verdict(7, event, verdict="NO TRADE", setup="NO_EDGE"),   # event day
    ]))
    con.execute("""INSERT INTO outcomes (drug_id, catalyst_date, ticker)
                   VALUES (7, ?, 'X')""", [event])
    got = outcomes.recent(con, lookback_days=45)
    assert len(got) == 1
    assert got["prior_verdict"].iloc[0] == "SHORT"


def test_workflow_commits_logs_and_guards_against_shrinking():
    wf = (ROOT / ".github" / "workflows" / "refresh.yml").read_text()
    assert "git add web/board.json logs/" in wf
    assert "Refuse to shrink the forward logs" in wf
