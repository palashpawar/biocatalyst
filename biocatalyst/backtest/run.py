"""The two backtests.

`run_runway`   tests the DILUTION_SHORT hypothesis as a clean point-in-time
               factor study anchored on 10-Q/10-K filing dates. Nothing about
               the anchor or the features can be revised after the fact.

`run_catalyst` tests whether the engine's setup labels separate outcomes
               around trial readouts. Anchored on ClinicalTrials.gov primary
               completion dates, which sponsors revise, so its timing is
               optimistic. Reported separately and never merged with the above.

Not testable here: VOL_SELL_RICH and VOL_BUY_CHEAP. Both compare a live option
straddle to a base rate, and there is no free source of historical option
chains. Those two setups remain unvalidated hypotheses.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .. import baserates
from ..config import RUNWAY_CRITICAL_MONTHS, RUNWAY_WARN_MONTHS
from ..sources import financing
from . import engine, pit, stats, universe

UNTESTABLE = ("VOL_SELL_RICH", "VOL_BUY_CHEAP")


def _runway_bucket(m) -> str:
    if pd.isna(m):
        return "unknown"
    if m < RUNWAY_CRITICAL_MONTHS:
        return f"a. <{RUNWAY_CRITICAL_MONTHS:.0f}mo"
    if m < RUNWAY_WARN_MONTHS:
        return f"b. {RUNWAY_CRITICAL_MONTHS:.0f}-{RUNWAY_WARN_MONTHS:.0f}mo"
    if m < 24:
        return "c. 12-24mo"
    return "d. >24mo"


def _size_bucket(mcap) -> str:
    if pd.isna(mcap):
        return "unknown"
    if mcap < 300e6:
        return "micro <$300M"
    if mcap < 2e9:
        return "small $0.3-2B"
    return "mid+ >$2B"


def run_runway(start: dt.date, end: dt.date, tickers: list[str],
               horizons=(21, 63, 126), verbose: bool = True) -> dict:
    """Point-in-time runway factor study anchored on filing dates."""
    if verbose:
        print(f"[runway] building filing events for {len(tickers)} tickers ...",
              flush=True)
    events = universe.filing_events(tickers, start, end, verbose=verbose)
    if events.empty:
        return {"summary": pd.DataFrame(), "events": events}
    if verbose:
        print(f"[runway] {len(events)} filings, "
              f"{events.ticker.nunique()} tickers", flush=True)

    closes, _, missing = engine.build_price_panel(
        list(events.ticker.unique()), start, end, verbose=verbose)
    rets = engine.event_returns(events, closes, horizons=horizons,
                                entry_offset=0)
    if rets.empty:
        return {"summary": pd.DataFrame(), "events": rets}

    if verbose:
        print(f"[runway] attaching point-in-time financials ...", flush=True)
    feats = []
    for t, grp in rets.groupby("ticker"):
        facts = pit.companyfacts(t)
        if not facts:
            continue
        for _, r in grp.iterrows():
            fin = pit.pit_financials(t, r["entry_date"], facts=facts)
            if not fin:
                continue
            # Restate shares on today's post-split basis: price history is
            # back-adjusted, so a raw historical count inflates market cap by
            # the cumulative split factor.
            shares = pit.shares_in_current_terms(
                fin.get("shares_outstanding"), t, r["entry_date"])
            mcap = shares * r["entry_price"] if shares else np.nan
            feats.append({**r.to_dict(),
                          "runway_months": fin["runway_months"],
                          "cash_usd": fin["cash_usd"],
                          "market_cap": mcap})
    df = pd.DataFrame(feats)
    if df.empty:
        return {"summary": pd.DataFrame(), "events": df}

    df["runway_bucket"] = df["runway_months"].map(_runway_bucket)
    df["size_bucket"] = df["market_cap"].map(_size_bucket)
    df["distressed_micro"] = (
        (df["runway_months"] < RUNWAY_CRITICAL_MONTHS)
        & (df["market_cap"] < 300e6)).map({True: "distressed micro",
                                           False: "everything else"})

    return {
        "events": df,
        "by_runway": stats.summarize(df, "runway_bucket", horizons=horizons),
        "by_size": stats.summarize(df, "size_bucket", horizons=horizons),
        "by_setup": stats.summarize(df, "distressed_micro", horizons=horizons),
        "missing_tickers": missing,
    }


def run_catalyst(start: dt.date, end: dt.date, horizons=(1, 5, 21, 63),
                 verbose: bool = True) -> dict:
    """Event study around trial readouts, bucketed by stage and prior."""
    if verbose:
        print("[catalyst] pulling historical readouts ...", flush=True)
    events = universe.catalyst_events(start, end, verbose=verbose)
    if events.empty:
        return {"events": events}
    if verbose:
        print(f"[catalyst] {len(events)} events, "
              f"{events.ticker.nunique()} tickers", flush=True)

    closes, _, missing = engine.build_price_panel(
        list(events.ticker.unique()), start, end, verbose=verbose)
    df = engine.event_returns(events, closes, horizons=horizons)
    if df.empty:
        return {"events": df}

    meta = events.set_index("nct_id")[["indication", "simplified_stage"]]
    df = df.drop(columns=["indication", "simplified_stage"], errors="ignore")
    df = df.join(meta, on="nct_id")

    df["loa"] = df.apply(
        lambda r: baserates.loa(r["simplified_stage"], r["indication"]), axis=1)
    df["prior_bucket"] = pd.cut(df["loa"], [0, .15, .35, 1.0],
                                labels=["low prior", "mid prior", "high prior"])
    df["runup_bucket"] = pd.cut(
        df["ret_20d"], [-np.inf, -.15, .15, np.inf],
        labels=["sold off", "flat", "ran up >15%"])

    return {
        "events": df,
        "by_stage": stats.summarize(df, "simplified_stage", horizons=horizons),
        "by_prior": stats.summarize(df, "prior_bucket", horizons=horizons),
        "by_runup": stats.summarize(df, "runup_bucket", horizons=horizons),
        "overall": stats.summarize(df, None, horizons=horizons),
        "missing_tickers": missing,
    }

def _burst_bucket(b) -> str:
    if pd.isna(b):
        return "unknown"
    if b < 0.75:
        return "a. quiet <0.75x"
    if b < 1.5:
        return "b. normal 0.75-1.5x"
    if b < 3.0:
        return "c. elevated 1.5-3x"
    return "d. burst >3x"


def _count_bucket(n) -> str:
    if pd.isna(n):
        return "unknown"
    n = int(n)
    if n == 0:
        return "a. none"
    if n <= 2:
        return "b. 1-2"
    if n <= 5:
        return "c. 3-5"
    return "d. 6+"


def run_cadence(start: dt.date, end: dt.date, tickers: list[str],
                horizons=(21, 63, 126), verbose: bool = True) -> dict:
    """Does 8-K filing activity predict anything?

    Anchored on 10-Q/10-K filing dates and measured only from 8-Ks already
    filed, so this is fully point-in-time -- unlike headline sentiment, which
    has no free historical archive. The interaction with cash runway is the
    question that matters: runway under six months is already a validated
    signal, so filing activity has to add something on top of it to be worth
    anything.
    """
    if verbose:
        print(f"[cadence] filing events for {len(tickers)} tickers ...",
              flush=True)
    events = universe.filing_events(tickers, start, end, verbose=verbose)
    if events.empty:
        return {"events": events}
    if verbose:
        print(f"[cadence] {len(events)} filings, "
              f"{events.ticker.nunique()} tickers", flush=True)

    closes, _, missing = engine.build_price_panel(
        list(events.ticker.unique()), start, end, verbose=verbose)
    rets = engine.event_returns(events, closes, horizons=horizons,
                                entry_offset=0)
    if rets.empty:
        return {"events": rets}

    if verbose:
        print("[cadence] attaching point-in-time 8-K history ...", flush=True)
    rows = []
    for t, grp in rets.groupby("ticker"):
        eightk = pit.submissions(t, since=start - dt.timedelta(days=400))
        facts = pit.companyfacts(t)
        for _, r in grp.iterrows():
            cad = pit.cadence_at(eightk, r["entry_date"])
            fin = pit.pit_financials(t, r["entry_date"], facts=facts) if facts else None
            rows.append({**r.to_dict(), **cad,
                         "runway_months": (fin or {}).get("runway_months")})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"events": df}

    df["burst_bucket"] = df["burst"].map(_burst_bucket)
    df["count_bucket"] = df["eightk_90d"].map(_count_bucket)

    # Does filing activity add anything on top of the validated runway signal?
    low = df["runway_months"] < RUNWAY_CRITICAL_MONTHS
    hot = df["burst"] >= 1.5
    df["interaction"] = np.where(
        low & hot, "low runway + burst",
        np.where(low & ~hot, "low runway only",
                 np.where(~low & hot, "burst only", "neither")))

    return {
        "events": df,
        "by_burst": stats.summarize(df, "burst_bucket", horizons=horizons),
        "by_count": stats.summarize(df, "count_bucket", horizons=horizons),
        "by_interaction": stats.summarize(df, "interaction", horizons=horizons),
        "missing_tickers": missing,
    }


def _dtc_bucket(v) -> str:
    if pd.isna(v):
        return "unknown"
    if v < 1.5:
        return "a. <1.5 days"
    if v < 3:
        return "b. 1.5-3 days"
    if v < 6:
        return "c. 3-6 days"
    return "d. 6+ days"


def run_squeeze(start: dt.date, end: dt.date, tickers: list[str],
                horizons=(21, 63, 126), verbose: bool = True) -> dict:
    """Does short crowding change what happens to a low-runway name?

    Short interest is the one risk input here with free history, so unlike
    headline sentiment it can be tested. Readings are matched by publication
    date, not settlement date -- FINRA discloses roughly eight business days
    late, and using the settlement date would let the study act on a position
    nobody could see yet.
    """
    if verbose:
        print(f"[squeeze] filing events for {len(tickers)} tickers ...",
              flush=True)
    events = universe.filing_events(tickers, start, end, verbose=verbose)
    if events.empty:
        return {"events": events}

    closes, _, missing = engine.build_price_panel(
        list(events.ticker.unique()), start, end, verbose=verbose)
    rets = engine.event_returns(events, closes, horizons=horizons,
                                entry_offset=0)
    if rets.empty:
        return {"events": rets}

    if verbose:
        print("[squeeze] attaching point-in-time short interest ...", flush=True)
    rows = []
    for i, (t, grp) in enumerate(rets.groupby("ticker"), 1):
        si_rows = pit.short_interest_history(t)
        facts = pit.companyfacts(t)
        for _, r in grp.iterrows():
            si = pit.short_interest_at(si_rows, r["entry_date"])
            fin = pit.pit_financials(t, r["entry_date"], facts=facts) if facts else None
            rows.append({**r.to_dict(),
                         "days_to_cover": (si or {}).get("days_to_cover"),
                         "shares_short": (si or {}).get("shares_short"),
                         "runway_months": (fin or {}).get("runway_months")})
        if verbose and i % 50 == 0:
            print(f"    short interest {i}/{rets.ticker.nunique()}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"events": df}

    df["dtc_bucket"] = df["days_to_cover"].map(_dtc_bucket)
    low = df["runway_months"] < RUNWAY_CRITICAL_MONTHS
    crowded = df["days_to_cover"] >= 5
    df["interaction"] = np.where(
        low & crowded, "low runway + crowded short",
        np.where(low & ~crowded, "low runway, uncrowded",
                 np.where(~low & crowded, "crowded short only", "neither")))

    return {
        "events": df,
        "by_dtc": stats.summarize(df, "dtc_bucket", horizons=horizons),
        "by_interaction": stats.summarize(df, "interaction", horizons=horizons),
        "missing_tickers": missing,
    }


def run_financing(start: dt.date, end: dt.date, tickers: list[str],
                  horizons=(21, 63, 126), verbose: bool = True) -> dict:
    """Does shelf/offering posture add anything to cash runway alone?

    Free in request terms -- it reads the same cached submissions index the
    8-K study uses -- and point-in-time by construction, since filing dates
    are immutable.
    """
    if verbose:
        print(f"[financing] filing events for {len(tickers)} tickers ...",
              flush=True)
    events = universe.filing_events(tickers, start, end, verbose=verbose)
    if events.empty:
        return {"events": events}

    closes, _, missing = engine.build_price_panel(
        list(events.ticker.unique()), start, end, verbose=verbose)
    rets = engine.event_returns(events, closes, horizons=horizons,
                                entry_offset=0)
    if rets.empty:
        return {"events": rets}

    if verbose:
        print("[financing] attaching shelf/offering history ...", flush=True)
    rows = []
    for t, grp in rets.groupby("ticker"):
        filings = pit.all_filings(t)
        facts = pit.companyfacts(t)
        for _, r in grp.iterrows():
            fz = financing.financing_at(filings, r["entry_date"])
            fin = pit.pit_financials(t, r["entry_date"], facts=facts) if facts else None
            runway = (fin or {}).get("runway_months")
            rows.append({**r.to_dict(), **fz,
                         "runway_months": runway,
                         **financing.dilution_readiness(fz, runway)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"events": df}

    df["readiness_bucket"] = df["dilution_label"]
    df["recent_deal"] = np.where(
        df["days_since_offering"].notna() & (df["days_since_offering"] <= 120),
        "priced <120d ago", "no recent deal")

    low = df["runway_months"] < RUNWAY_CRITICAL_MONTHS
    serial = df["offerings_24m"].fillna(0) >= 2
    df["interaction"] = np.where(
        low & serial, "low runway + serial issuer",
        np.where(low & ~serial, "low runway, rare issuer",
                 np.where(~low & serial, "serial issuer only", "neither")))

    return {
        "events": df,
        "by_readiness": stats.summarize(df, "readiness_bucket", horizons=horizons),
        "by_recent_deal": stats.summarize(df, "recent_deal", horizons=horizons),
        "by_interaction": stats.summarize(df, "interaction", horizons=horizons),
        "missing_tickers": missing,
    }
