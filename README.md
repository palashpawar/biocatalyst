# biocatalyst

A local catalyst-event engine for biotech and pharma. It builds a forward
calendar of binary events (trial readouts, PDUFA dates), enriches each one with
cash runway, crowding, option pricing and published approval base rates, and
ranks them as **named setups** rather than blanket buy/sell calls.

```bash
python3 -m biocatalyst.cli refresh      # pull every source into DuckDB (~3 min)
python3 -m biocatalyst.cli board --why 5  # ranked board + reasoning
python3 -m dashboard.app                # http://127.0.0.1:5057
                                        # sorted soonest-first;
                                        # tickers link to Robinhood
```

## Why it does not just say "long" or "short"

The direction of a binary readout is not forecastable from public data. If it
were, the option market would not price 50%+ straddles into these dates. What
*is* tractable, and what this engine actually measures:

Each setup maps to a plain verdict shown on the board. **SHORT** and **LONG**
are directional; **SELL VOL** and **BUY VOL** are not — they are a view on the
price of the event, not on which way it resolves. Every verdict is shown next
to its evidence tier, so a `supported` call and an `untested` one never look
alike.

| Setup | Verdict | What it keys on |
|---|---|---|
| `DILUTION_SHORT` | SHORT | Cash runs out around the catalyst. Financing is near-certain and gets priced into strength. |
| `VOL_SELL_RICH` | SELL VOL | The straddle prices the event well above what that event type historically delivers. |
| `VOL_BUY_CHEAP` | BUY VOL | The straddle prices it below the historical analogue. |
| `RUNUP_FADE` | SHORT | Crowded into a dated binary on a weak prior. |
| `BASE_RATE_LONG` | LONG | Strong prior, funded past the event, not yet crowded. |
| `NO_EDGE` | NO TRADE | The honest default. |

## Data sources

| Source | Use | Auth |
|---|---|---|
| ClinicalTrials.gov v2 | Readout dates, trial design, enrollment | none |
| SEC EDGAR XBRL | Cash, burn, shares → runway | UA header |
| SEC EDGAR full-text | PDUFA dates from 8-K filings | UA header |
| Yahoo (yfinance) | Prices, realized vol, option chains | none |
| BiopharmCatalyst | Curated calendar (optional) | see below |

### On BiopharmCatalyst

BPC is the best curated catalyst calendar available, and the original reason
this project exists. Its free tier returns **only the first 10 rows**: every
other row arrives with `blur: true` and its catalyst fields (stage, date, NCT)
nulled out, while the numeric fields their subscription sells stay populated.

Harvesting those blurred rows would be circumventing a paywall, so this project
does not. `sources/bpc.py` whitelists free-tier fields and drops everything in
`PAID_FIELDS` before it reaches the database. Instead, `sources/discovery.py`
reconstructs an equivalent calendar from the primary sources BPC itself
aggregates — roughly 350 catalysts over a 180-day window, versus BPC's 10.

If you subscribe, you can use what you are entitled to:

```bash
export BIOCATALYST_BPC_COOKIE='<your session cookie>'
python3 -m biocatalyst.cli refresh --source bpc --include-paid
```


## Backtest harness

```bash
python3 -m biocatalyst.cli backtest --mode both --start 2021-01-01 --end 2025-06-30 --save
```

Two studies, deliberately kept apart because their anchors are not equally trustworthy.

**Runway factor study** (`--mode runway`) tests the `DILUTION_SHORT` hypothesis.
Anchored on 10-Q/10-K **filing dates**, which are immutable and public the day
they happen. Every feature is rebuilt from EDGAR facts filtered by their `filed`
date, so a quarter that ended in September is invisible until it is filed in
November, and later restatements of old periods never leak backwards.

**Catalyst event study** (`--mode catalyst`) tests whether the setup labels
separate outcomes around trial readouts. Anchored on ClinicalTrials.gov primary
completion dates — and those are served **only at their current value**. There
is no public version history (the v2 endpoint is 404, the internal one 403), and
sponsors revise these dates, often repeatedly. A date used here may not have
been knowable in advance, so this study is **optimistic about timing**. It is
never merged with the runway results.

### What the harness does to avoid fooling itself

| Guard | Why |
|---|---|
| Point-in-time EDGAR (`filed <=` as-of) | A period end is not a knowledge date. Without this, every observation leaks the future. |
| Abnormal returns vs XBI | Biotech moves together; raw returns would mostly measure the sector's drift over the sample. |
| Ticker-clustered bootstrap | One company contributes many events. Treating them as independent inflates n and shrinks intervals. |
| Bootstrap, not t-tests | Readout returns are violently fat-tailed; the normal approximation does not hold. |
| Bonferroni across all buckets | Dozens of buckets are tested at once. Nominal 5% means nothing at that width. |
| `MIN_CLUSTERS = 12` | Buckets below this are printed but never called a result. |
| Delisted tickers counted | Companies that were acquired or went to zero have no price history. The count is reported as the survivorship bias, not hidden. |
| Split-consistent market caps | Price history is back-adjusted for splits; EDGAR share counts are not. Multiplying them raw inflated 20% of observations — one by 53,000,000× — and flipped the sign of every size result. Historical share counts are restated onto today's basis. |

Significance is flagged in three tiers: `*** survives multiple testing`,
`* nominal only, fails multiple testing`, and `thin (n_tickers=N)`.

#### What it found

Over 2021-01-01 to 2025-06-30, 5,379 point-in-time filing observations, 338 tickers:

- **Cash runway under six months underperforms**: −5.4% abnormal at 21 days
  (p=0.000) and −16.2% at 126 days (p=0.002), monotone across runway buckets.
  Both survive multiple testing.
- **Heavy 8-K filing activity compounds it.** 6+ 8-Ks in the prior 90 days is
  itself −13.0% at 126 days (p=0.000). Combined with low runway and a microcap
  valuation it reaches **−39.8% at 126 days on a 19% hit rate**. The *burst
  ratio* (activity against a company's own baseline) found nothing — it is the
  absolute count that carries the signal.
- **Microcaps underperform**: −10.5% at 126 days, while large caps run +8.8%.
  Both survive multiple testing. An earlier version of this README claimed the
  opposite; that analysis used corrupted market caps (see the guard table).
- **Run-up fade is not supported** (−1.5% @1d, p=0.26), and readout direction
  is unforecastable (+0.11% @1d across all readouts).

Caveat worth respecting: the low-runway-plus-busy result was negative in 2021,
2022, 2023 and 2024 but **reversed positive in the 2025 sub-sample** (+9.3%,
n=14). Thin, but it is the most recent evidence and it does not confirm.

## What cannot be tested

`VOL_SELL_RICH` and `VOL_BUY_CHEAP` both compare a live straddle to a base rate,
and there is no free source of historical option chains. Those two setups remain
**unvalidated hypotheses**, and the harness says so on every run.

## Honest limitations

- **Weights are judgement calls, not fitted parameters.** Every number in
  `score.py` and `baserates.py` was chosen by hand. The backtest harness tests
  whether the *setups* separate outcomes; it does not tune the weights, and
  nothing here is optimised against the sample.
- **Two of the six setups cannot be tested at all** (see above).
- **Base rates are population averages.** BIO/Informa/QLS 2011–2020, 12,728
  phase transitions. They say nothing about a specific asset.
- **Discovery is uncurated.** A ClinicalTrials.gov primary completion date is
  when the last patient hits the endpoint, *not* when the company announces.
  Real readouts typically trail it by one to two quarters. BPC's value is that
  an analyst has done this reconciliation; the discovery calendar has not.
- **Sponsor→ticker mapping is fuzzy.** Name-matched against SEC's registry;
  subsidiaries and foreign listings get dropped or mismatched.
- **Option-implied moves use the first expiry on/after the catalyst**, which can
  sit well past it, diluting the event premium.

Treat the board as a research queue: it tells you where to look, not what to do.

## Layout

```
biocatalyst/
  baserates.py        published LOA priors, therapeutic-area classifier
  features.py         joins raw tables, derives runway/crowding/move ratios
  score.py            setup classification, conviction, stated reasons
  sources/
    discovery.py      catalyst calendar from ClinicalTrials.gov + EDGAR
    bpc.py            BiopharmCatalyst client, fuzzy date parser
    ctgov.py          trial design facts
    edgar.py          cash, burn, runway from XBRL
    prices.py         price/vol history, option-implied moves
  backtest/
    pit.py            point-in-time financials (filed-date filtered)
    universe.py       historical event universes, both anchors
    engine.py         price panel, abnormal returns vs XBI
    stats.py          ticker-clustered bootstrap, multiple-testing control
    run.py            the two studies
dashboard/app.py      local Flask board
```

Run the tests with `python3 -m pytest tests/ -q`.
