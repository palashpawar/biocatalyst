"""Event-study machinery: price panel, abnormal returns, feature assembly."""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

BENCHMARK = "XBI"          # SPDR S&P Biotech, equal-weighted small/mid biotech
DEFAULT_HORIZONS = (1, 5, 21, 63)
ENTRY_OFFSET = 5           # trading days before the event we could act


def build_price_panel(tickers: list[str], start: dt.date, end: dt.date,
                      batch: int = 100, verbose: bool = False):
    """Adjusted closes for the universe plus the benchmark.

    Returns (closes, volumes, missing). `missing` is the set of tickers with no
    usable history -- mostly companies that were acquired or delisted, which is
    the survivorship bias in this study made explicit rather than hidden.
    """
    uniq = sorted({t.upper() for t in tickers if t} | {BENCHMARK})
    close_parts, vol_parts, missing = [], [], set()

    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        try:
            data = yf.download(chunk, start=start - dt.timedelta(days=10),
                               end=end + dt.timedelta(days=120),
                               auto_adjust=True, progress=False,
                               group_by="ticker", threads=True)
        except Exception:
            missing.update(chunk)
            continue
        for t in chunk:
            try:
                sub = data[t] if len(chunk) > 1 else data
                c = sub["Close"].dropna()
                if len(c) < 30:
                    missing.add(t)
                    continue
                close_parts.append(c.rename(t))
                vol_parts.append(sub["Volume"].rename(t))
            except Exception:
                missing.add(t)
        if verbose:
            print(f"    prices {min(i + batch, len(uniq))}/{len(uniq)}", flush=True)

    closes = pd.concat(close_parts, axis=1) if close_parts else pd.DataFrame()
    vols = pd.concat(vol_parts, axis=1) if vol_parts else pd.DataFrame()
    if not closes.empty:
        closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
        vols.index = closes.index
    return closes, vols, missing


def _pos(index: pd.DatetimeIndex, day: dt.date) -> int | None:
    """Index position of the first trading day on or after `day`."""
    ts = pd.Timestamp(day)
    loc = index.searchsorted(ts)
    return int(loc) if loc < len(index) else None


def event_returns(events: pd.DataFrame, closes: pd.DataFrame,
                  horizons=DEFAULT_HORIZONS,
                  entry_offset: int = ENTRY_OFFSET) -> pd.DataFrame:
    """Abnormal returns from entry to each horizon, benchmarked to XBI.

    Abnormal, not raw: biotech moves together, and an unadjusted study would
    mostly measure the sector's drift over the sample window.
    """
    if events.empty or closes.empty or BENCHMARK not in closes.columns:
        return pd.DataFrame()

    idx = closes.index
    bench = closes[BENCHMARK]
    out = []

    for _, e in events.iterrows():
        t = e["ticker"]
        if t not in closes.columns:
            continue
        px = closes[t]
        at = _pos(idx, e["event_date"])
        if at is None:
            continue
        entry = at - entry_offset
        if entry < 0 or entry >= len(idx):
            continue

        p0, b0 = px.iloc[entry], bench.iloc[entry]
        if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(b0) or b0 <= 0:
            continue

        rec = {
            "ticker": t,
            "event_date": e["event_date"],
            "entry_date": idx[entry].date(),
            "anchor": e.get("anchor"),
            "nct_id": e.get("nct_id"),
            "simplified_stage": e.get("simplified_stage"),
            "indication": e.get("indication"),
            "entry_price": float(p0),
        }

        # Realized vol and run-up measured strictly before entry.
        hist = px.iloc[max(0, entry - 63):entry + 1]
        if len(hist) > 21:
            r = np.log(hist / hist.shift(1)).dropna()
            rec["rvol_60d"] = float(r.std() * np.sqrt(252))
            rec["ret_20d"] = float(hist.iloc[-1] / hist.iloc[-21] - 1)
        else:
            rec["rvol_60d"] = rec["ret_20d"] = np.nan

        for h in horizons:
            j = entry + entry_offset + h
            if j >= len(idx):
                rec[f"abn_{h}d"] = np.nan
                continue
            p1, b1 = px.iloc[j], bench.iloc[j]
            if not np.isfinite(p1) or not np.isfinite(b1) or b1 <= 0:
                rec[f"abn_{h}d"] = np.nan
                continue
            rec[f"abn_{h}d"] = float((p1 / p0 - 1) - (b1 / b0 - 1))

        out.append(rec)

    return pd.DataFrame(out)
