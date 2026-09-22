"""Form 4 insider transactions.

Only two transaction codes carry information about what an insider chose to
do with their own money:

    P  open-market purchase
    S  open-market sale

Everything else is compensation plumbing. Across one quarter of SEC data,
grants (A), option exercises (M) and share-withholding for taxes (F) are 74%
of all rows -- and an F is an automatic disposal to cover a tax bill, not a
decision to sell. Counting them would turn every vesting date into a bearish
signal.

Two access paths, because they answer different questions:

  * `recent_activity` parses Form 4 XML directly. Filed within two business
    days, so it is the freshest signal in the whole project.
  * `load_bulk_quarter` reads SEC's quarterly Form 345 data sets, which is the
    only tractable way to get years of this for a backtest.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import zipfile

import pandas as pd

from ..config import DATA_DIR
from ..http import get
from .edgar import ticker_to_cik

BULK_URL = ("https://www.sec.gov/files/structureddata/data/"
            "insider-transactions-data-sets/{quarter}_form345.zip")
CACHE = DATA_DIR / "cache"

OPEN_MARKET = {"P", "S"}


def _archive(cik: int, accession: str) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}")


# Insiders are not interchangeable. A CEO or CFO trades with a view of the
# whole company; a director sees board packets; a 10% holder may be a fund
# rebalancing for reasons that have nothing to do with the business. Rank is
# why openinsider.com surfaces a Title column, and it is in the filing too.
SENIOR = re.compile(r"\b(chief exec|ceo|chief financial|cfo|president|"
                    r"chief operating|coo|chief medical|cmo|chief scientific|cso)\b",
                    re.I)


def _owner_role(xml: str) -> tuple[str, str]:
    """(role, title) for the reporting owner."""
    def flag(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.S)
        v = (m.group(1).strip().lower() if m else "")
        return v in ("1", "true")

    t = re.search(r"<officerTitle>(.*?)</officerTitle>", xml, re.S)
    title = re.sub(r"\s+", " ", t.group(1)).strip() if t else ""
    if flag("isOfficer"):
        return ("senior" if SENIOR.search(title) else "officer"), title
    if flag("isDirector"):
        return "director", title or "Director"
    if flag("isTenPercentOwner"):
        return "ten_percent", title or "10% owner"
    return "other", title


def _parse_form4(xml: str) -> list[dict]:
    """Pull open-market transactions out of one Form 4."""
    role, title = _owner_role(xml)
    owner = re.search(r"<rptOwnerName>(.*?)</rptOwnerName>", xml, re.S)
    owner = re.sub(r"\s+", " ", owner.group(1)).strip() if owner else ""
    out = []
    blocks = re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
                        xml, re.S)
    for b in blocks:
        code = re.search(r"<transactionCode>(.*?)</transactionCode>", b)
        code = code.group(1).strip() if code else None
        if code not in OPEN_MARKET:
            continue
        shares = re.search(r"<transactionShares>\s*<value>(.*?)</value>", b, re.S)
        price = re.search(r"<transactionPricePerShare>\s*<value>(.*?)</value>", b, re.S)
        date = re.search(r"<transactionDate>\s*<value>(.*?)</value>", b, re.S)
        try:
            n = float(shares.group(1).strip()) if shares else 0.0
        except ValueError:
            continue
        try:
            px = float(price.group(1).strip()) if price else None
        except ValueError:
            px = None
        # Share count after the trade lets us express size as a fraction of
        # what the insider held. Selling 5,000 of 240,000 is housekeeping;
        # selling 5,000 of 6,000 is an exit.
        after = re.search(
            r"<sharesOwnedFollowingTransaction>\s*<value>(.*?)</value>", b, re.S)
        try:
            held_after = float(after.group(1).strip()) if after else None
        except ValueError:
            held_after = None
        delta_own = None
        if held_after is not None and n:
            before = held_after - n if code == "P" else held_after + n
            if before > 0:
                delta_own = (n / before) * (1 if code == "P" else -1)

        out.append({"code": code, "shares": n, "price": px,
                    "date": date.group(1).strip() if date else None,
                    "role": role, "title": title, "owner": owner,
                    "held_after": held_after, "delta_own": delta_own})
    return out


def recent_activity(ticker: str, days: int = 90,
                    max_filings: int = 60) -> dict:
    """Net open-market insider activity over a trailing window."""
    cik = ticker_to_cik().get(ticker.upper())
    if not cik:
        return {}
    try:
        recent = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
                     ).json()["filings"]["recent"]
    except Exception:
        return {}

    cutoff = dt.date.today() - dt.timedelta(days=days)
    bought = sold = 0.0
    buy_usd = sell_usd = 0.0
    senior_usd = 0.0
    buyers, sellers = set(), set()
    biggest_move, biggest_desc = 0.0, None
    seen = 0

    for form, filed, acc, doc in zip(recent.get("form", []),
                                     recent.get("filingDate", []),
                                     recent.get("accessionNumber", []),
                                     recent.get("primaryDocument", [])):
        if form != "4" or seen >= max_filings:
            continue
        try:
            fd = dt.date.fromisoformat(filed)
        except ValueError:
            continue
        if fd < cutoff:
            break                       # the index is newest-first
        seen += 1
        # primaryDocument points at the XSL-rendered view; the raw XML that
        # actually carries the fields sits one directory up.
        raw = doc.split("/")[-1]
        try:
            xml = get(f"{_archive(cik, acc)}/{raw}").text
        except Exception:
            continue
        for tx in _parse_form4(xml):
            usd = (tx["shares"] * tx["price"]) if tx["price"] else 0.0
            if tx["code"] == "P":
                bought += tx["shares"]
                buy_usd += usd
                buyers.add(tx["owner"] or acc)
            else:
                sold += tx["shares"]
                sell_usd += usd
                sellers.add(tx["owner"] or acc)
            if tx["role"] == "senior":
                senior_usd += usd if tx["code"] == "P" else -usd
            d = tx.get("delta_own")
            if d is not None and abs(d) > abs(biggest_move):
                biggest_move = d
                biggest_desc = (f"{tx['owner'] or 'insider'} "
                                f"({tx['title'] or tx['role']}) "
                                f"{'+' if d > 0 else ''}{d:.0%} of holding")

    net_usd = buy_usd - sell_usd
    total_usd = buy_usd + sell_usd
    return {
        "ticker": ticker.upper(),
        "window_days": days,
        "form4_filings": seen,
        "shares_bought": bought,
        "shares_sold": sold,
        "buy_usd": buy_usd,
        "sell_usd": sell_usd,
        "net_usd": net_usd,
        # -1 (all selling) to +1 (all buying); 0 when nothing open-market happened.
        "insider_tilt": round(net_usd / total_usd, 3) if total_usd else 0.0,
        # Broken out because a C-suite trade and a 10%-holder rebalance are
        # not the same evidence, and an aggregate tilt hides the difference.
        "senior_net_usd": senior_usd,
        "n_buyers": len(buyers),
        "n_sellers": len(sellers),
        # A cluster of separate insiders buying is the classic strong read.
        "cluster_buy": len(buyers) >= 3,
        "biggest_delta_own": round(biggest_move, 4) if biggest_move else None,
        "biggest_move_desc": biggest_desc,
        "pulled_at": dt.datetime.now(),
    }


def fetch(tickers: list[str], days: int = 90, verbose: bool = False) -> pd.DataFrame:
    rows = []
    uniq = sorted({t.upper() for t in tickers if t})
    for i, t in enumerate(uniq, 1):
        rec = recent_activity(t, days=days)
        if rec:
            rows.append(rec)
        if verbose and i % 25 == 0:
            print(f"    insider {i}/{len(uniq)}", flush=True)
    return pd.DataFrame(rows)


def load_bulk_quarter(quarter: str) -> pd.DataFrame:
    """One quarter of SEC Form 345 data, cached.

    `quarter` looks like "2025q2". Returns one row per open-market
    transaction, keyed by the FILING date -- Form 4 is due within two business
    days, but the filing date is when the market could actually see it.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"form345_{quarter}.zip"
    if not path.exists():
        try:
            resp = get(BULK_URL.format(quarter=quarter))
            path.write_bytes(resp.content)
        except Exception:
            return pd.DataFrame()

    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        path.unlink(missing_ok=True)
        return pd.DataFrame()

    with z.open("SUBMISSION.tsv") as f:
        sub = {}
        for r in csv.DictReader(io.TextIOWrapper(f, "utf-8"), delimiter="\t"):
            if r.get("DOCUMENT_TYPE") != "4":
                continue
            sub[r["ACCESSION_NUMBER"]] = (r.get("ISSUERTRADINGSYMBOL"),
                                          r.get("FILING_DATE"))

    rows = []
    with z.open("NONDERIV_TRANS.tsv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, "utf-8"), delimiter="\t"):
            if r.get("TRANS_CODE") not in OPEN_MARKET:
                continue
            meta = sub.get(r["ACCESSION_NUMBER"])
            if not meta or not meta[0] or meta[0] == "NA":
                continue
            ticker, filed = meta
            try:
                filed_date = dt.datetime.strptime(filed, "%d-%b-%Y").date()
                shares = float(r.get("TRANS_SHARES") or 0)
                price = float(r.get("TRANS_PRICEPERSHARE") or 0) or None
            except (ValueError, TypeError):
                continue
            rows.append({"ticker": ticker.upper(), "filed_date": filed_date,
                         "accession": r["ACCESSION_NUMBER"],
                         "code": r["TRANS_CODE"], "shares": shares,
                         "price": price,
                         "usd": shares * price if price else 0.0})
    return pd.DataFrame(rows)
