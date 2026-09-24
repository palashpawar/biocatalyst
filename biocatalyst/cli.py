"""Command line entry point: refresh the database, print the catalyst board."""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

from . import export as exporter
from . import logs
from . import features, score
from .backtest import run as bt
from .backtest import stats as btstats
from .backtest import universe as btuniverse
from .db import connect, upsert
from .sources import (bpc, ctgov, discovery, edgar, financing,
                      guidance as guidance_src, insider, news,
                      prices, shortinterest)


def cmd_refresh(args) -> int:
    if args.source == "bpc":
        print("[1/10] BiopharmCatalyst FDA calendar ...", flush=True)
        cal = bpc.fetch(max_pages=args.max_pages, include_paid=args.include_paid)
        if not args.include_paid and len(cal) <= 10:
            print("      NOTE: BPC's free tier returns only the first 10 rows; "
                  "the rest arrive blurred. Use --source discovery for full "
                  "coverage, or --include-paid with a subscriber cookie.")
    else:
        print("[1/10] Catalyst discovery (ClinicalTrials.gov + EDGAR) ...", flush=True)
        cal = discovery.build_calendar(horizon_days=args.horizon,
                                       with_pdufa=not args.no_pdufa,
                                       verbose=True,
                                       lookback_days=args.lookback)
    if cal.empty:
        print("  no catalysts returned; aborting", file=sys.stderr)
        return 1
    print(f"      {len(cal)} catalysts, {cal.ticker.nunique()} tickers")

    tickers = [t for t in cal["ticker"].dropna().unique()]
    ncts = [n for n in cal["nct_id"].dropna().unique()]
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
        ncts = ncts[:args.max_tickers]

    # Each stage is persisted as it lands: a stall in a later source then
    # costs only that source, not the whole run.
    # The feed only returns upcoming catalysts, so rebuild just that slice and
    # keep everything already in the past. Deleting the whole table each run
    # threw away the history needed to see what a catalyst actually did.
    with connect() as con:
        # CI starts from an empty database. Seed the forward logs from the
        # committed CSVs first, or tonight's appends would have nothing to
        # append to and every accumulated verdict would be lost.
        seeded = logs.load(con)
        print(f"      seeded logs: {seeded}")
        # Rebuild the forward slice and the lookback window; anything older
        # stays as history.
        con.execute(
            "DELETE FROM catalysts WHERE catalyst_date_hi >= current_date - INTERVAL (?) DAY",
            [args.lookback])
        upsert(con, "catalysts", cal)

    print(f"[2/10] ClinicalTrials.gov ({len(ncts)} NCTs) ...", flush=True)
    trials = ctgov.fetch(ncts)
    print(f"      {len(trials)} trials")
    with connect() as con:
        upsert(con, "trials", trials)

    print(f"[3/10] SEC EDGAR cash & burn ({len(tickers)} tickers) ...", flush=True)
    fins = edgar.fetch(tickers, verbose=True)
    print(f"      {len(fins)} filers")
    with connect() as con:
        upsert(con, "financials", fins)

    print(f"[4/10] Prices & volatility ({len(tickers)} tickers) ...", flush=True)
    px = prices.fetch(tickers, verbose=True)
    print(f"      {len(px)} price histories")

    # Market cap from EDGAR shares x last close, avoiding a slow per-ticker call.
    if not px.empty and not fins.empty:
        from .backtest.pit import shares_in_current_terms
        today = dt.date.today()
        shares = fins.set_index("ticker")["shares_outstanding"]
        px["market_cap"] = [
            (lambda s: s * row["last"] if s else None)(
                shares_in_current_terms(shares.get(row["ticker"]),
                                        row["ticker"], today))
            for _, row in px.iterrows()]
    with connect() as con:
        upsert(con, "prices", px)

    if not args.no_news:
        print(f"[5/10] News sentiment & filing cadence ({len(tickers)}) ...",
              flush=True)
        companies = (cal.dropna(subset=["ticker"])
                     .drop_duplicates("ticker")
                     .set_index("ticker")["company_name"].to_dict())
        sent = news.fetch(tickers, companies, verbose=True)
        print(f"      {len(sent)} tickers scored")
        with connect() as con:
            n_sent = logs.append(con, "sentiment", sent)
        print(f"      {n_sent} sentiment changes logged")
    else:
        sent = pd.DataFrame()

    print("[6/10] Option-implied catalyst moves ...", flush=True)
    today = dt.date.today()
    # Don't gate on the `optionable` flag: the discovery source can't know it.
    # implied_move() returns None for names with no listed chain, which is the
    # same answer at roughly the same cost.
    near = cal[cal["catalyst_date_lo"].notna()].copy()
    # Restrict to names with working price history. Delisted tickers make
    # yfinance retry internally and can stall the whole step.
    if not px.empty:
        near = near[near["ticker"].isin(set(px["ticker"]))]
    near = near[near["catalyst_date_lo"].map(
        lambda d: d is not None and 0 <= (d - today).days <= args.implied_horizon)]
    if args.max_tickers:
        near = near.head(args.max_tickers)

    imp_rows = []
    seen = set()
    pairs = []
    for _, r in near.iterrows():
        key = (r["ticker"], r["catalyst_date_lo"])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    print(f"      pricing {len(pairs)} chains", flush=True)
    for i, (tick, cdate) in enumerate(pairs, 1):
        rec = prices.implied_move(tick, cdate)
        if rec:
            imp_rows.append(rec)
        if i % 25 == 0:
            print(f"    implied {i}/{len(pairs)}", flush=True)
    imp = pd.DataFrame(imp_rows)
    print(f"      {len(imp)} option chains priced")

    with connect() as con:
        upsert(con, "implied", imp)
    if not args.no_short_interest:
        print(f"[7/10] FINRA short interest ({len(tickers)}) ...", flush=True)
        si = shortinterest.fetch(tickers, verbose=True)
        print(f"      {len(si)} readings")
        with connect() as con:
            upsert(con, "short_interest", si)
    else:
        si = pd.DataFrame()

    print(f"[8/10] Shelf, offerings & insider activity ({len(tickers)}) ...",
          flush=True)
    from .backtest.pit import all_filings
    fin_rows, ins_rows = [], []
    fin_by_ticker = (fins.set_index("ticker")["runway_months"].to_dict()
                     if not fins.empty else {})
    for i, t in enumerate(tickers, 1):
        fz = financing.financing_at(all_filings(t))
        fz.update(financing.dilution_readiness(fz, fin_by_ticker.get(t)))
        fin_rows.append({"ticker": t, **fz, "pulled_at": dt.datetime.now()})
        if not args.no_insider:
            rec = insider.recent_activity(t)
            if rec:
                ins_rows.append({**rec, "snapshot_date": dt.date.today()})
        if i % 25 == 0:
            print(f"    financing {i}/{len(tickers)}", flush=True)
    fin_df = pd.DataFrame(fin_rows).drop(
        columns=["shelf_age_days"], errors="ignore")
    ins_df = pd.DataFrame(ins_rows)
    print(f"      {len(fin_df)} financing, {len(ins_df)} insider")
    with connect() as con:
        con.execute("DELETE FROM financing")
        upsert(con, "financing", fin_df)
        upsert(con, "insider", ins_df)

    print("[9/10] Readout guidance from 8-K text ...", flush=True)
    guide = guidance_src.fetch(verbose=True) if not args.no_guidance else pd.DataFrame()
    print(f"      {len(guide)} companies with stated timing")
    with connect() as con:
        con.execute("DELETE FROM guidance")
        upsert(con, "guidance", guide)

    print("[10/10] Scoring board & realised outcomes ...", flush=True)
    from . import outcomes as outcomes_mod
    with connect() as con:
        feat = features.build(con, horizon_days=args.horizon)
        board = score.score(feat) if not feat.empty else pd.DataFrame()
        snaps = outcomes_mod.snapshot_verdicts(con, board)
        n_verd = logs.append(con, "verdict_log", snaps)
        outs = outcomes_mod.compute(con, lookback_days=args.lookback, verbose=True)
        upsert(con, "outcomes", outs)
        written = logs.dump(con)
    print(f"      {n_verd} of {len(snaps)} verdicts changed and were logged; "
          f"{len(outs)} outcomes scored")
    print(f"      logs on disk: {written}")

    print(f"\nwrote catalysts={len(cal)} trials={len(trials)} "
          f"financials={len(fins)} prices={len(px)} implied={len(imp)} "
          f"sentiment={len(sent)} short={len(si)} financing={len(fin_df)} "
          f"insider={len(ins_df)} guidance={len(guide)} outcomes={len(outs)}")
    return 0


def cmd_board(args) -> int:
    with connect(read_only=True) as con:
        feat = features.build(con, horizon_days=args.horizon)
    if feat.empty:
        print("No catalysts in window. Run `refresh` first.")
        return 0

    board = score.score(feat)
    if args.sort == "date":
        board = board.sort_values(["days_to_catalyst", "conviction"],
                                  ascending=[True, False])
    if args.setup:
        board = board[board["setup"] == args.setup.upper()]
    if args.min_conviction:
        board = board[board["conviction"] >= args.min_conviction]
    board = board.head(args.limit)

    cols = ["verdict", "ticker", "catalyst_date_raw", "days_to_catalyst",
            "stage", "setup", "conviction", "evidence", "loa",
            "implied_move", "expected_move", "runway_months"]
    view = board[cols].copy()
    view["loa"] = view["loa"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    for c in ("implied_move", "expected_move"):
        view[c] = view[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    view["runway_months"] = view["runway_months"].map(
        lambda v: f"{v:.0f}" if pd.notna(v) else "-")

    print(view.to_string(index=False))
    if args.why:
        print("\n" + "=" * 70)
        for _, r in board.head(args.why).iterrows():
            print(f"\n{r['ticker']}  {r['drug_name']}")
            arrow = "" if r["directional"] else "  (not directional)"
            print(f"  VERDICT: {r['verdict']}{arrow}")
            print(f"  {r['setup']} ({r['conviction']}) -- {r['setup_desc']}")
            print(f"  evidence: {r['evidence']} -- {r['evidence_note']}")
            print(f"  {r['indication']} | {r['stage']} | {r['catalyst_date_raw']}")
            print(f"  why: {r['reasons']}")
    return 0


def cmd_backtest(args) -> int:
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    print(f"Backtest window {start} -> {end}\n")

    print("Deriving the biotech universe from historical trial sponsors ...",
          flush=True)
    cat_events = btuniverse.catalyst_events(start, end, verbose=False)
    tickers = sorted(cat_events["ticker"].dropna().unique()) if not cat_events.empty else []
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    print(f"  {len(tickers)} tickers\n")

    n_tests = 0

    if args.mode in ("runway", "both"):
        res = bt.run_runway(start, end, tickers, verbose=True)
        print("\n" + "=" * 100)
        print("RUNWAY FACTOR STUDY  (anchor: 10-Q/10-K filing dates -- "
              "point-in-time, no revision risk)")
        print("=" * 100)
        ev = res.get("events")
        if ev is None or ev.empty:
            print("no observations")
        else:
            print(f"{len(ev)} filing observations, {ev.ticker.nunique()} tickers, "
                  f"{len(res.get('missing_tickers', []))} tickers without price "
                  f"history (delisted/acquired -- survivorship bias)\n")
            keys = (("by_runway", "By cash runway at filing"),
                    ("by_size", "By market cap at filing"),
                    ("by_setup", "DILUTION_SHORT condition"))
            tables = [res.get(k) for k, _ in keys]
            n_tests += btstats.apply_multiple_testing(tables)
            for (key, title), tbl in zip(keys, tables):
                if tbl is not None and not tbl.empty:
                    print(f"\n-- {title}")
                    print(btstats.format_table(tbl))
            if args.save:
                ev.to_csv("data/backtest_runway.csv", index=False)
                print("\nsaved data/backtest_runway.csv")

    if args.mode in ("cadence", "both"):
        res = bt.run_cadence(start, end, tickers, verbose=True)
        print("\n" + "=" * 100)
        print("8-K CADENCE STUDY  (anchor: 10-Q/10-K filing dates -- "
              "point-in-time; 8-K dates are immutable)")
        print("=" * 100)
        ev = res.get("events")
        if ev is None or ev.empty:
            print("no observations")
        else:
            print(f"{len(ev)} filing observations, {ev.ticker.nunique()} tickers\n")
            keys = (("by_burst", "By 8-K burst (90d vs own trailing-year rate)"),
                    ("by_count", "By raw 8-K count in prior 90 days"),
                    ("by_interaction", "Interaction with the validated runway signal"))
            tables = [res.get(k) for k, _ in keys]
            n_tests += btstats.apply_multiple_testing(tables)
            for (key, title), tbl in zip(keys, tables):
                if tbl is not None and not tbl.empty:
                    print(f"\n-- {title}")
                    print(btstats.format_table(tbl))
            if args.save:
                ev.to_csv("data/backtest_cadence.csv", index=False)
                print("\nsaved data/backtest_cadence.csv")

    if args.mode in ("insider", "both"):
        res = bt.run_insider(start, end, tickers, verbose=True)
        print("\n" + "=" * 100)
        print("INSIDER STUDY  (anchor: filing dates; Form 4 matched by FILING "
              "date, open-market P/S only)")
        print("=" * 100)
        ev = res.get("events")
        if ev is None or ev.empty:
            print("no observations")
        else:
            print(f"{len(ev)} observations, {ev.ticker.nunique()} tickers\n")
            keys = (("by_net", "By net open-market insider flow (90d)"),
                    ("by_cluster", "By cluster buying"),
                    ("by_interaction", "Insider buying against the runway signal"))
            tables = [res.get(k) for k, _ in keys]
            n_tests += btstats.apply_multiple_testing(tables)
            for (key, title), tbl in zip(keys, tables):
                if tbl is not None and not tbl.empty:
                    print(f"\n-- {title}")
                    print(btstats.format_table(tbl))
            if args.save:
                ev.to_csv("data/backtest_insider.csv", index=False)

    if args.mode in ("financing", "both"):
        res = bt.run_financing(start, end, tickers, verbose=True)
        print("\n" + "=" * 100)
        print("FINANCING STUDY  (anchor: filing dates; shelf and offering "
              "history, immutable filing dates)")
        print("=" * 100)
        ev = res.get("events")
        if ev is None or ev.empty:
            print("no observations")
        else:
            print(f"{len(ev)} observations, {ev.ticker.nunique()} tickers\n")
            keys = (("by_readiness", "By dilution readiness"),
                    ("by_recent_deal", "By recency of the last offering"),
                    ("by_interaction", "Serial issuance against the runway signal"))
            tables = [res.get(k) for k, _ in keys]
            n_tests += btstats.apply_multiple_testing(tables)
            for (key, title), tbl in zip(keys, tables):
                if tbl is not None and not tbl.empty:
                    print(f"\n-- {title}")
                    print(btstats.format_table(tbl))
            if args.save:
                ev.to_csv("data/backtest_financing.csv", index=False)

    if args.mode in ("squeeze", "both"):
        res = bt.run_squeeze(start, end, tickers, verbose=True)
        print("\n" + "=" * 100)
        print("SQUEEZE STUDY  (anchor: filing dates; short interest matched by "
              "FINRA PUBLICATION date, not settlement)")
        print("=" * 100)
        ev = res.get("events")
        if ev is None or ev.empty:
            print("no observations")
        else:
            print(f"{len(ev)} observations, {ev.ticker.nunique()} tickers\n")
            keys = (("by_dtc", "By days to cover"),
                    ("by_interaction", "Crowding against the validated runway signal"))
            tables = [res.get(k) for k, _ in keys]
            n_tests += btstats.apply_multiple_testing(tables)
            for (key, title), tbl in zip(keys, tables):
                if tbl is not None and not tbl.empty:
                    print(f"\n-- {title}")
                    print(btstats.format_table(tbl))
            if args.save:
                ev.to_csv("data/backtest_squeeze.csv", index=False)
                print("\nsaved data/backtest_squeeze.csv")

    if args.mode in ("catalyst", "both"):
        res = bt.run_catalyst(start, end, verbose=True)
        print("\n" + "=" * 100)
        print("CATALYST EVENT STUDY  (anchor: ClinicalTrials.gov primary "
              "completion -- dates are as-revised, timing is OPTIMISTIC)")
        print("=" * 100)
        ev = res.get("events")
        if ev is None or ev.empty:
            print("no observations")
        else:
            print(f"{len(ev)} events, {ev.ticker.nunique()} tickers, "
                  f"{len(res.get('missing_tickers', []))} tickers without price "
                  f"history\n")
            keys = (("overall", "All readouts"),
                    ("by_stage", "By phase"),
                    ("by_prior", "By base-rate prior"),
                    ("by_runup", "By 20-day run-up into the date"),
                    ("by_design", "By trial design"),
                    ("by_enrollment", "By enrollment"))
            tables = [res.get(k) for k, _ in keys]
            n_tests += btstats.apply_multiple_testing(tables)
            for (key, title), tbl in zip(keys, tables):
                if tbl is not None and not tbl.empty:
                    print(f"\n-- {title}")
                    print(btstats.format_table(tbl))
            if args.save:
                ev.to_csv("data/backtest_catalyst.csv", index=False)
                print("\nsaved data/backtest_catalyst.csv")

    print("\n" + "=" * 100)
    print(btstats.bonferroni_note(n_tests))
    print("NOT TESTED: " + ", ".join(bt.UNTESTABLE) +
          " -- both compare a live straddle to a base rate, and there is no "
          "free source of historical option chains.")
    print("=" * 100)
    return 0


def cmd_export(args) -> int:
    path = exporter.export(args.out, horizon=args.horizon,
                           lookback=args.lookback)
    size = path.stat().st_size
    import json as _json
    payload = _json.loads(path.read_text())
    print(f"wrote {path} ({size / 1024:.0f} KB, "
          f"{len(payload['rows'])} upcoming, "
          f"{len(payload.get('recent', []))} recent)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="biocatalyst")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="pull every source into DuckDB")
    r.add_argument("--source", choices=["discovery", "bpc"], default="discovery",
                   help="discovery = primary sources (default); bpc = BPC calendar")
    r.add_argument("--horizon", type=int, default=180)
    r.add_argument("--no-pdufa", action="store_true")
    r.add_argument("--max-pages", type=int, default=None)
    r.add_argument("--max-tickers", type=int, default=None)
    r.add_argument("--implied-horizon", type=int, default=90)
    r.add_argument("--lookback", type=int, default=45,
                   help="days of already-passed catalysts to keep and score")
    r.add_argument("--no-guidance", action="store_true",
                   help="skip the 8-K readout-guidance search")
    r.add_argument("--no-insider", action="store_true",
                   help="skip the Form 4 pull (the slowest step)")
    r.add_argument("--no-short-interest", action="store_true",
                   help="skip the FINRA short-interest pull")
    r.add_argument("--no-news", action="store_true",
                   help="skip the news/sentiment pull")
    r.add_argument("--include-paid", action="store_true",
                   help="use BPC subscriber fields (requires BIOCATALYST_BPC_COOKIE)")
    r.set_defaults(func=cmd_refresh)

    b = sub.add_parser("board", help="ranked catalyst board")
    b.add_argument("--horizon", type=int, default=180)
    b.add_argument("--sort", choices=["date", "conviction"], default="date")
    b.add_argument("--limit", type=int, default=30)
    b.add_argument("--setup", type=str, default=None)
    b.add_argument("--min-conviction", type=int, default=0)
    b.add_argument("--why", type=int, default=0, metavar="N",
                   help="print reasoning for the top N rows")
    b.set_defaults(func=cmd_board)

    k = sub.add_parser("backtest", help="point-in-time backtest of the setups")
    k.add_argument("--start", default="2021-01-01")
    k.add_argument("--end", default="2025-06-30")
    k.add_argument("--mode",
                   choices=["runway", "catalyst", "cadence", "squeeze",
                            "financing", "insider", "both"],
                   default="both")
    k.add_argument("--max-tickers", type=int, default=None)
    k.add_argument("--save", action="store_true", help="write per-event CSVs")
    k.set_defaults(func=cmd_backtest)

    e = sub.add_parser("export", help="write web/board.json for static hosting")
    e.add_argument("--out", default=None)
    e.add_argument("--horizon", type=int, default=180)
    e.add_argument("--lookback", type=int, default=45)
    e.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
