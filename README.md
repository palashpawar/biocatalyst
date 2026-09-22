# biocatalyst

A catalyst-event engine for biotech and pharma. It builds a calendar of binary
events (trial readouts, PDUFA dates), enriches each one with cash runway, short
crowding, filing activity, option pricing, news sentiment and published approval
base rates, and ranks them as **named setups** rather than blanket buy/sell calls.

Every setup carries the result of an actual backtest, so a call with evidence
behind it never looks like one without.

**Live board:** https://biocatalyst-iota.vercel.app

```bash
python3 -m biocatalyst.cli refresh        # pull every source into DuckDB
python3 -m biocatalyst.cli board --why 5  # ranked board + reasoning
python3 -m biocatalyst.cli backtest --mode both --save
python3 -m biocatalyst.cli export         # write web/board.json for hosting
python3 -m dashboard.app                  # local Flask board, :5057
```

## Why it does not just say "long" or "short"

The direction of a binary readout is not forecastable from public data. If it
were, the option market would not price 50%+ straddles into these dates. What
*is* tractable, and what this engine actually measures:

| Setup | Verdict | Keys on | Evidence |
|---|---|---|---|
| `DILUTION_SHORT` | SHORT | Cash runs out around the catalyst; financing gets priced into strength | **supported** |
| `VOL_SELL_RICH` | SELL VOL | Straddle prices the event above its historical analogue | untested |
| `VOL_BUY_CHEAP` | BUY VOL | Straddle prices it below | untested |
| `RUNUP_FADE` | SHORT | Crowded into a dated binary on a weak prior | **not supported** |
| `BASE_RATE_LONG` | LONG | Strong prior, funded past the event, not yet crowded | weak |
| `NO_EDGE` | NO TRADE | The honest default | — |

`SELL VOL` and `BUY VOL` are not directional. They are a view on the price of
the event, not on which way it resolves.

## What the backtests found

2021-01-01 to 2025-06-30, 5,379 point-in-time filing observations, 338 tickers.
Abnormal returns against XBI, ticker-clustered bootstrap, Bonferroni across
every bucket tested.

- **Cash runway under six months underperforms.** −5.4% at 21 days (p=0.000),
  −16.2% at 126 days (p=0.002), monotone across runway buckets.
- **Heavy 8-K filing compounds it.** 6+ 8-Ks in the prior 90 days is −13.0% at
  126 days (p=0.000) on its own. With low runway and a microcap valuation it
  reaches **−39.8% at 126 days on a 19% hit rate**. The *burst ratio* against a
  company's own baseline found nothing — the absolute count carries the signal.
- **Serial issuance is the actual mechanism.** Splitting low-runway names on
  two offerings in 24 months: serial issuers ran −8.8% at 21 days and **−27.7%
  at 126 days** (both survive); rare issuers ran −0.3% (p=0.87) and +2.2%
  (p=0.64) — nothing. Low cash only predicts a decline when the company
  habitually raises. Dilution readiness is monotone across its four buckets,
  with "loaded" at −17.0% at 126 days, and a company that priced a deal in the
  last 120 days ran −9.0% at 126 days.
- **Short crowding erases the edge.** Splitting low-runway names on 5 days to
  cover: uncrowded ran −7.2% at 21 days and −20.2% at 126 days (both survive);
  crowded ran −1.1% (p=0.63) and −6.4% (p=0.23). Crowded shorts on *well-funded*
  names drifted the other way, +3.1% at 63 days (p=0.001) — the squeeze itself.
- **Microcaps underperform**: −10.5% at 126 days; large caps +8.8%. Both survive.
- **Run-up fade is not supported** (−1.5% @1d, p=0.26), and readout direction is
  unforecastable (+0.11% @1d across all readouts).

Caveat worth respecting: the low-runway-plus-busy result was negative in 2021
through 2024 but **reversed positive in the 2025 sub-sample** (+9.3%, n=14).
Thin, but it is the most recent evidence and it does not confirm.

## Recent results

The board keeps catalysts for 45 days after they happen and scores what the
stock did — day-one move, move since, and abnormal move against XBI, measured
the same way as the backtest so the numbers are comparable.

Verdicts are snapshotted on every refresh into `verdict_log`. An outcome joins
the last snapshot taken *before* its catalyst date, so the engine is graded on
what it said in advance rather than on a verdict re-derived with the answer
visible. That column reads "not yet logged" for catalysts that passed before
logging began, and fills in from there.

Only day- and part-of-month-precision dates are scored: a readout stated as
"Q4 2026" has no event day, so a move around an arbitrary date in that range is
noise.

## Data sources

| Source | Use | Auth |
|---|---|---|
| ClinicalTrials.gov v2 | Readout dates, trial design, enrollment | none |
| SEC EDGAR XBRL | Cash, burn, shares → runway | UA header |
| SEC EDGAR full-text | PDUFA dates, drug and indication from 8-Ks | UA header |
| SEC EDGAR submissions | 8-K filing cadence | UA header |
| SEC EDGAR submissions | Shelf (S-3) and offering (424B5) history | UA header |
| SEC Form 4 / bulk Form 345 | Open-market insider buys and sales, officer rank, ownership change | UA header |
| FINRA | Consolidated short interest, ~9 years of history | none |
| Yahoo (yfinance) | Prices, realized vol, option chains | none |
| Google News RSS | Headlines for sentiment | none |
| BiopharmCatalyst | Curated calendar (optional) | see below |

### On BiopharmCatalyst

BPC is the best curated catalyst calendar available, and the original reason
this project exists. Its free tier returns **only the first 10 rows**: every
other row arrives with `blur: true` and its catalyst fields (stage, date, NCT)
nulled out, while the numeric fields their subscription sells stay populated.

Harvesting those blurred rows would be circumventing a paywall, so this project
does not. `sources/bpc.py` whitelists free-tier fields and drops everything in
`PAID_FIELDS` before it reaches the database. Instead `sources/discovery.py`
reconstructs an equivalent calendar from the primary sources BPC itself
aggregates — roughly 350 catalysts over a 180-day window, versus BPC's 10.

If you subscribe, you can use what you are entitled to:

```bash
export BIOCATALYST_BPC_COOKIE='<your session cookie>'
python3 -m biocatalyst.cli refresh --source bpc --include-paid
```

## Backtest harness

```bash
python3 -m biocatalyst.cli backtest --mode both --start 2021-01-01 --save
```

Five studies (`--mode runway | cadence | squeeze | financing | catalyst | both`), kept apart
because their anchors are not equally trustworthy.

**Filing-anchored** (`runway`, `cadence`, `squeeze`) use 10-Q/10-K filing dates,
which are immutable and public the day they happen. Every feature is rebuilt
from EDGAR facts filtered by their `filed` date, so a quarter ending in
September is invisible until it is filed in November, and later restatements of
old periods never leak backwards.

**Catalyst-anchored** (`catalyst`) uses ClinicalTrials.gov primary completion
dates, served **only at their current value**. There is no public version
history (the v2 endpoint is 404, the internal one 403) and sponsors revise these
dates, so that study is **optimistic about timing** and is never merged with the
filing-anchored results.

### What the harness does to avoid fooling itself

| Guard | Why |
|---|---|
| Point-in-time EDGAR (`filed <=` as-of) | A period end is not a knowledge date. Without this every observation leaks the future. |
| Short interest by publication date | FINRA discloses ~8 business days after settlement; keying off settlement acts on a position nobody could see. |
| Abnormal returns vs XBI | Biotech moves together; raw returns would mostly measure the sector's drift. |
| Ticker-clustered bootstrap | One company contributes many events. Treating them as independent inflates n and shrinks intervals. |
| Bootstrap, not t-tests | Readout returns are violently fat-tailed; the normal approximation does not hold. |
| Bonferroni across all buckets | Dozens of buckets tested at once. Nominal 5% means nothing at that width. |
| `MIN_CLUSTERS = 12` | Buckets below this are printed but never called a result. |
| Split-consistent market caps | Price history is back-adjusted for splits; EDGAR share counts are not. Multiplying them raw inflated 20% of observations — one by 53,000,000× — and flipped the sign of every size result. |
| Delisted tickers counted | Companies acquired or gone to zero have no price history. The count is reported as survivorship bias, not hidden. |

Significance is flagged in three tiers: `*** survives multiple testing`,
`* nominal only, fails multiple testing`, and `thin (n_tickers=N)`.

## What cannot be tested

- **`VOL_SELL_RICH` / `VOL_BUY_CHEAP`** compare a live straddle to a base rate,
  and there is no free source of historical option chains. Permanently
  unvalidated on free data; the harness says so on every run.
- **Insider activity** is shipped but **not yet backtested**. It is parsed
  from Form 4 and filtered to open-market `P` and `S`, with officer rank and
  the change in the insider's own holding — openinsider.com surfaces the same
  fields and is a good manual cross-check, but everything it shows is in the
  filings, so this reads the source rather than scraping a third party.
  Contradictions are surfaced as CAUTION chips on a short without moving the
  conviction number. The bulk Form 345
  data sets make it testable (`insider.load_bulk_quarter`), which is the next
  thing to run; until then it is shown and never scored.
- **News sentiment** has no free point-in-time archive either, so it is
  **context only and never moves a verdict**. Scores are logged daily to the
  `sentiment` table so that it becomes testable once enough forward history
  accumulates. X/Twitter is not used at all: the search API is paid, and
  StockTwits' self-applied tags are gamed (13 of 13 tagged messages on one
  sample read "Bullish", including one saying the stock was down).

## Honest limitations

- **Weights are judgement calls, not fitted parameters.** Every number in
  `score.py` and `baserates.py` was chosen by hand. The harness tests whether
  the *setups* separate outcomes; it does not tune weights, and nothing is
  optimised against the sample.
- **Base rates are population averages.** BIO/Informa/QLS 2011–2020, 12,728
  phase transitions. They say nothing about a specific asset.
- **Discovery is uncurated.** A ClinicalTrials.gov primary completion date is
  when the last patient hits the endpoint, *not* when the company announces.
  Real readouts typically trail it by one to two quarters. BPC's value is that
  an analyst has done that reconciliation; this calendar has not. It is the
  largest remaining accuracy gap.
- **Short interest is bi-monthly and stale.** Up to two weeks old before the
  next reading lands, so the crowding shown may not be the crowding on the day.
- **PDUFA rows often lack an indication.** The 8-K extractor pulls the date
  reliably and the drug less so; where the indication is missing the row falls
  back to the flat `pdufa` base rate with no therapeutic-area adjustment.
- **Sponsor→ticker mapping is fuzzy.** Name-matched against SEC's registry;
  subsidiaries and foreign listings get dropped or mismatched.
- **Option-implied moves use the first expiry on/after the catalyst**, which can
  sit well past it, diluting the event premium.

Treat the board as a research queue: it tells you where to look, not what to do.
Nothing here is investment advice.

## Layout

```
biocatalyst/
  config.py           paths, rate limits, scoring thresholds
  db.py               DuckDB schema and upserts
  http.py             rate-limited session with backoff
  baserates.py        published LOA priors, therapeutic-area classifier
  features.py         joins raw tables, derives runway/crowding/move ratios
  score.py            setup classification, verdicts, conviction, reasons
  outcomes.py         realised moves for catalysts that have passed
  export.py           static board.json for hosting
  sources/
    discovery.py      catalyst calendar from ClinicalTrials.gov + EDGAR
    bpc.py            BiopharmCatalyst client, fuzzy date parser
    ctgov.py          trial design facts
    edgar.py          cash, burn, runway from XBRL (TTL-cached)
    prices.py         price/vol history, option-implied moves
    shortinterest.py  FINRA short interest, squeeze metrics
    financing.py      shelf registrations, offerings, dilution readiness
    insider.py        Form 4 open-market buys/sales (P and S only)
    news.py           headline sentiment, 8-K filing cadence
  backtest/
    pit.py            point-in-time financials, 8-K and short-interest history
    universe.py       historical event universes, both anchors
    engine.py         price panel, abnormal returns vs XBI
    stats.py          ticker-clustered bootstrap, multiple-testing control
    run.py            the five studies
dashboard/app.py      local Flask board
web/                  static site published to Vercel
.github/workflows/    nightly refresh
```

Deployment and the nightly refresh are documented in [DEPLOY.md](DEPLOY.md).

Run the tests with `python3 -m pytest tests/ -q` (129 tests).
