"""Headline sentiment and filing cadence.

Scope note: this is **context, not signal**. Nothing here feeds a verdict or
moves conviction. Free news sources return only recent items -- there is no
free point-in-time archive -- so a sentiment rule could not be run through the
backtest harness before being trusted, and the closest thing that *was* tested
(RUNUP_FADE, crowding into a dated binary) came out unsupported.

So sentiment is displayed and snapshotted daily into the `sentiment` table.
Once enough forward history accumulates, it becomes testable the same way the
runway factor was, and only then should it be allowed to change a verdict.

Filing cadence is different: 8-K dates are immutable and already historical,
so that column is backtestable today.
"""
from __future__ import annotations

import datetime as dt
import re
import urllib.parse
import warnings
import xml.etree.ElementTree as ET

import pandas as pd

from ..http import get
from .edgar import ticker_to_cik

warnings.filterwarnings("ignore")

GOOGLE_NEWS = "https://news.google.com/rss/search"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Multi-word phrases carry most of the meaning in biotech headlines, and they
# are checked before single words so "did not meet primary endpoint" is not
# scored positively off the word "meet".
NEGATIVE_PHRASES = [
    "complete response letter", "refuse to file", "clinical hold",
    "did not meet", "failed to meet", "fails to meet", "missed the primary",
    "did not achieve", "failed to achieve", "no significant difference",
    "discontinu", "terminat", "halted", "warning letter", "going concern",
    "reverse split", "reverse stock split", "delisting", "delist",
    "serious adverse", "safety signal", "patient death", "adverse event",
    "public offering", "registered direct", "at-the-market", "dilut",
    "downgrade", "cut to", "withdraw", "recall", "subpoena", "lawsuit",
    "restructuring", "workforce reduction", "layoff", "misses",
]
POSITIVE_PHRASES = [
    "met the primary", "met primary", "meets primary", "achieved the primary",
    "achieves primary", "statistically significant", "breakthrough therapy",
    "fast track", "orphan drug", "priority review", "accelerated approval",
    "positive topline", "positive results", "positive data", "fda approval",
    "fda approves", "approved by the fda", "granted approval",
    "regenerative medicine advanced", "upgrade", "raised to", "beats",
    "well tolerated", "favorable safety", "licensing agreement",
    "strategic partnership", "milestone payment", "buyout", "acquisition of",
]
NEGATIVE_WORDS = ["fail", "failure", "reject", "rejected", "decline", "drop",
                  "plunge", "slump", "risk", "concern", "delay", "delayed",
                  "setback", "loss", "weak", "disappointing", "investigation"]
POSITIVE_WORDS = ["approval", "approved", "positive", "success", "successful",
                  "surge", "soar", "rally", "beat", "strong", "promising",
                  "encouraging", "grant", "granted", "expansion", "win"]

_NEGATORS = re.compile(r"\b(?:not|no|never|without|fails? to|failed to|"
                       r"did ?n[o']t|does ?n[o']t|unable to)\b")


def _windows_negated(text: str, at: int, back: int = 32) -> bool:
    """Is there a negator shortly before this position?"""
    return bool(_NEGATORS.search(text[max(0, at - back):at]))


def score_text(text: str) -> tuple[int, int]:
    """Count positive and negative hits in one headline."""
    t = (text or "").lower()
    pos = neg = 0
    for phrase in NEGATIVE_PHRASES:
        neg += t.count(phrase)
    for phrase in POSITIVE_PHRASES:
        for m in re.finditer(re.escape(phrase), t):
            # "not approved" must not count as positive.
            if _windows_negated(t, m.start()):
                neg += 1
            else:
                pos += 1
    for word in NEGATIVE_WORDS:
        for m in re.finditer(rf"\b{re.escape(word)}", t):
            neg += 1
    for word in POSITIVE_WORDS:
        for m in re.finditer(rf"\b{re.escape(word)}", t):
            if _windows_negated(t, m.start()):
                neg += 1
            else:
                pos += 1
    return pos, neg


def _parse_rss(xml_text: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        when = None
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                when = dt.datetime.strptime(pub, fmt).date()
                break
            except ValueError:
                continue
        if title:
            out.append({"title": title, "published": when, "link": link})
    return out


def google_news(ticker: str, company: str | None = None,
                days: int = 21) -> list[dict]:
    """Recent headlines for a ticker. Free, no key, roughly 100 items."""
    query = f'"{company}"' if company else f"{ticker} stock"
    url = f"{GOOGLE_NEWS}?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        items = _parse_rss(get(url).text)
    except Exception:
        return []
    cutoff = dt.date.today() - dt.timedelta(days=days)
    return [i for i in items if i["published"] and i["published"] >= cutoff]


def yahoo_news(ticker: str, days: int = 21) -> list[dict]:
    """Headlines from yfinance, which are curated to the ticker."""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    cutoff = dt.date.today() - dt.timedelta(days=days)
    out = []
    for n in raw:
        c = n.get("content", n)
        title = c.get("title")
        when = None
        stamp = c.get("pubDate") or c.get("providerPublishTime")
        if isinstance(stamp, str):
            try:
                when = dt.datetime.fromisoformat(
                    stamp.replace("Z", "+00:00")).date()
            except ValueError:
                when = None
        elif isinstance(stamp, (int, float)):
            when = dt.datetime.fromtimestamp(stamp).date()
        if title and when and when >= cutoff:
            out.append({"title": title, "published": when,
                        "link": c.get("canonicalUrl", {}).get("url", "")})
    return out


def filing_cadence(ticker: str, asof: dt.date | None = None) -> dict:
    """8-K filing counts in trailing windows.

    Unlike headlines, filing dates are immutable and fully historical, so this
    column can go through the backtest harness today.
    """
    asof = asof or dt.date.today()
    cik = ticker_to_cik().get(ticker.upper())
    if not cik:
        return {}
    try:
        recent = get(SUBMISSIONS.format(cik=cik)).json()["filings"]["recent"]
    except Exception:
        return {}

    dates = []
    for form, filed in zip(recent.get("form", []), recent.get("filingDate", [])):
        if not str(form).startswith("8-K"):
            continue
        try:
            d = dt.date.fromisoformat(filed)
        except ValueError:
            continue
        if d <= asof:
            dates.append(d)
    if not dates:
        return {"eightk_30d": 0, "eightk_90d": 0, "eightk_365d": 0,
                "days_since_8k": None}

    dates.sort(reverse=True)
    return {
        "eightk_30d": sum(1 for d in dates if (asof - d).days <= 30),
        "eightk_90d": sum(1 for d in dates if (asof - d).days <= 90),
        "eightk_365d": sum(1 for d in dates if (asof - d).days <= 365),
        "days_since_8k": (asof - dates[0]).days,
    }


# Words too common to identify a company in a headline.
_COMMON_WORDS = {
    "american", "national", "international", "global", "united", "general",
    "new", "first", "next", "one", "open", "true", "core", "prime", "summit",
    "vertex", "alpha", "beta", "delta", "apex", "atlas", "nova", "orion",
}

_GENERIC = re.compile(
    r"\b(?:inc|corp|corporation|company|companies|co|ltd|limited|plc|"
    r"holdings?|group|sa|nv|ag|ab|"
    r"therapeutics?|pharmaceuticals?|pharma|biosciences?|biopharmaceuticals?|"
    r"biopharma|labs?|laboratories|technologies|sciences|medicines?|health|"
    r"bio|the|and)\b", re.I)


def company_tokens(ticker: str, company: str | None) -> set[str]:
    """Distinctive words that should appear in a headline about this company."""
    tokens = {ticker.lower()}
    if company:
        stripped = _GENERIC.sub(" ", company.lower())
        for w in re.findall(r"[a-z][a-z0-9'\-]{2,}", stripped):
            if w not in _COMMON_WORDS:
                tokens.add(w)
    return tokens


def is_relevant(title: str, tokens: set[str]) -> bool:
    """Does the headline actually name this company?

    Google News returns anything that *mentions* the query, so a Merck search
    surfaces "Novo Nordisk shares plunge". Scoring those attributes another
    company's news to this ticker, which is worse than having no sentiment.
    """
    t = (title or "").lower()
    return any(re.search(rf"\b{re.escape(tok)}\b", t) for tok in tokens)


def sentiment_for(ticker: str, company: str | None = None,
                  days: int = 21) -> dict:
    """Blend both headline sources into one recency-weighted score."""
    items = google_news(ticker, company, days) + yahoo_news(ticker, days)
    tokens = company_tokens(ticker, company)
    seen, unique, dropped = set(), [], 0
    for i in items:
        key = re.sub(r"\W+", "", (i["title"] or "").lower())[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        if not is_relevant(i["title"], tokens):
            dropped += 1
            continue
        unique.append(i)

    today = dt.date.today()
    num = den = 0.0
    pos_n = neg_n = 0
    worst = best = None
    worst_s = best_s = 0

    for i in unique:
        p, n = score_text(i["title"])
        if p == n == 0:
            continue
        net = p - n
        pos_n += p
        neg_n += n
        age = max(0, (today - i["published"]).days) if i["published"] else days
        weight = 0.5 ** (age / 7.0)   # one-week half-life
        num += net * weight
        den += (p + n) * weight
        if net < worst_s:
            worst_s, worst = net, i["title"]
        if net > best_s:
            best_s, best = net, i["title"]

    score = round(num / den, 3) if den else 0.0
    # A +1.00 built from two headlines is not a strong reading. Callers show
    # scored_articles alongside the score so thin samples are visible.
    thin = (pos_n + neg_n) < 4
    return {
        "ticker": ticker.upper(),
        "snapshot_date": today,
        "articles": len(unique),
        "scored_articles": pos_n + neg_n,
        "pos_hits": pos_n,
        "neg_hits": neg_n,
        "sentiment": score,
        "thin": bool(thin),
        "top_positive": (best or "")[:220] or None,
        "top_negative": (worst or "")[:220] or None,
        "pulled_at": dt.datetime.now(),
    }


def fetch(tickers: list[str], companies: dict[str, str] | None = None,
          days: int = 21, verbose: bool = False) -> pd.DataFrame:
    companies = companies or {}
    rows = []
    uniq = sorted({t.upper() for t in tickers if t})
    for i, t in enumerate(uniq, 1):
        rec = sentiment_for(t, companies.get(t), days)
        cadence = filing_cadence(t)
        rows.append({**rec, **cadence})
        if verbose and i % 25 == 0:
            print(f"    news {i}/{len(uniq)}", flush=True)
    df = pd.DataFrame(rows)
    # `dropped_offtopic` is diagnostic only and is not persisted.
    return df.drop(columns=["dropped_offtopic"], errors="ignore")
