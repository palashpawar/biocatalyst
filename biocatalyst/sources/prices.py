"""Price history, realized volatility, and option-implied catalyst moves."""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

TRADING_DAYS = 252


def _stats_from_close(close: pd.Series, vol: pd.Series) -> dict:
    close = close.dropna()
    if len(close) < 25:
        return {}
    ret = np.log(close / close.shift(1)).dropna()
    dollar_vol = (close * vol.reindex(close.index)).dropna()

    def _chg(n: int):
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1) if len(close) > n else None

    return {
        "last": float(close.iloc[-1]),
        "ret_20d": _chg(20),
        "ret_60d": _chg(60),
        "rvol_20d": float(ret.tail(20).std() * np.sqrt(TRADING_DAYS)),
        "rvol_60d": float(ret.tail(60).std() * np.sqrt(TRADING_DAYS)),
        "adv_usd": float(dollar_vol.tail(20).mean()) if len(dollar_vol) else None,
        "pct_of_52w_high": float(close.iloc[-1] / close.tail(TRADING_DAYS).max()),
    }


def fetch(tickers: list[str], verbose: bool = False, batch: int = 100) -> pd.DataFrame:
    """Batch-download a year of history and derive price/vol/liquidity stats.

    Market cap is deliberately NOT taken from yfinance's per-ticker `info`
    endpoint -- that is one slow HTTP round trip per name. It is computed
    downstream as EDGAR shares outstanding x last close.
    """
    uniq = sorted({t.upper() for t in tickers if t})
    rows = []
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        try:
            data = yf.download(chunk, period="1y", auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for t in chunk:
            try:
                if len(chunk) == 1:
                    close, vol = data["Close"], data["Volume"]
                else:
                    if t not in data.columns.get_level_values(0):
                        continue
                    close, vol = data[t]["Close"], data[t]["Volume"]
                stat = _stats_from_close(close, vol)
                if stat:
                    rows.append({"ticker": t, **stat, "market_cap": None,
                                 "pulled_at": dt.datetime.now()})
            except Exception:
                continue
        if verbose:
            print(f"    prices {min(i + batch, len(uniq))}/{len(uniq)}", flush=True)
    return pd.DataFrame(rows)


def implied_move(ticker: str, catalyst_date: dt.date | None) -> dict | None:
    """ATM straddle-implied move for the first expiry on/after the catalyst.

    Returns the straddle cost as a fraction of spot, which is the market's
    price for the event. Comparing it to a base-rate move is the core of the
    volatility read: rich straddles favour selling premium, cheap ones buying.
    """
    if not catalyst_date:
        return None
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options or []
        if not expiries:
            return None
        future = [e for e in expiries if dt.date.fromisoformat(e) >= catalyst_date]
        if not future:
            return None
        expiry = future[0]

        chain = tk.option_chain(expiry)
        spot = tk.fast_info.get("lastPrice") or tk.history(period="1d")["Close"].iloc[-1]
        spot = float(spot)

        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return None

        c = calls.iloc[(calls["strike"] - spot).abs().argmin()]
        p = puts.iloc[(puts["strike"] - spot).abs().argmin()]

        def _mid(row):
            bid, ask = row.get("bid") or 0, row.get("ask") or 0
            return (bid + ask) / 2 if bid and ask else (row.get("lastPrice") or 0)

        straddle = _mid(c) + _mid(p)
        if straddle <= 0:
            return None

        atm_iv = float(np.nanmean([c.get("impliedVolatility"),
                                   p.get("impliedVolatility")]))
        return {
            "ticker": ticker,
            "catalyst_date": catalyst_date,
            "expiry": dt.date.fromisoformat(expiry),
            "atm_iv": atm_iv,
            "implied_move": straddle / spot,
            "straddle_pct": straddle / spot,
            "pulled_at": dt.datetime.now(),
        }
    except Exception:
        return None
