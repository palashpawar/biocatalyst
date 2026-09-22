"""Turn features into a named setup, a directional lean, and stated reasons.

Design stance: this does NOT emit a blanket BUY/SELL per ticker. The direction
of a binary readout is close to unforecastable from outside the blind -- the
repeatable structure in biotech sits in financing pressure, crowding into known
dates, and the price of event volatility. So the engine names the *setup* it
sees and shows its working. Every weight below is a judgement call.
"""
from __future__ import annotations

import pandas as pd

from .config import (IMPLIED_RICH_RATIO, RUNUP_HOT_20D,
                     RUNWAY_CRITICAL_MONTHS, RUNWAY_WARN_MONTHS)

# What the backtest actually found (2021-01-01 to 2025-06-30, filing-anchored
# point-in-time study, 5,379 observations across 338 tickers). See README.
SETUP_EVIDENCE = {
    "DILUTION_SHORT": ("supported",
                       "runway <6mo: -5.4% abn @21d (p=0.000), -16.2% @126d "
                       "(p=0.002), monotone across runway buckets. Sharpest "
                       "with 6+ 8-Ks in 90d AND microcap: -39.8% @126d "
                       "(p=0.000, 19% hit rate). Requires an UNCROWDED short: "
                       "split on 5 days to cover, uncrowded ran -7.2% @21d / "
                       "-20.2% @126d (both survive) while crowded ran -1.1% "
                       "(p=0.63). Driven by serial issuers: low runway + 2+ "
                       "offerings in 24mo ran -8.8% @21d / -27.7% @126d, while "
                       "rare issuers ran flat (p=0.87). Caveat: 2025 sub-sample "
                       "reversed positive"),
    "RUNUP_FADE": ("not supported",
                   "ran up >15% into a readout: -1.5% @1d (p=0.26), "
                   "+0.1% @21d -- no measurable edge"),
    "BASE_RATE_LONG": ("weak",
                       "phase 3 readouts +1.6% @21d nominal (p=0.033) but "
                       "fails multiple testing across 77 buckets"),
    "VOL_SELL_RICH": ("untested", "no free source of historical option chains"),
    "VOL_BUY_CHEAP": ("untested", "no free source of historical option chains"),
    "NO_EDGE": ("n/a", "the default"),
}

# The plain-language call for each setup. The two volatility setups are not
# directional: they are a view on the price of the event, not on which way it
# resolves, so they are labelled as such rather than dressed up as long/short.
VERDICT = {
    "DILUTION_SHORT": ("SHORT", True),
    "RUNUP_FADE": ("SHORT", True),
    "BASE_RATE_LONG": ("LONG", True),
    "VOL_SELL_RICH": ("SELL VOL", False),
    "VOL_BUY_CHEAP": ("BUY VOL", False),
    "NO_EDGE": ("NO TRADE", False),
}

SETUPS = {
    "DILUTION_SHORT": "Cash runs out around the catalyst; financing likely into strength",
    "VOL_SELL_RICH": "Options price the event well above its historical analogue",
    "VOL_BUY_CHEAP": "Options price the event below its historical analogue",
    "RUNUP_FADE": "Crowded into a dated binary with a weak prior",
    "BASE_RATE_LONG": "Strong prior, funded, not yet crowded",
    "NO_EDGE": "Nothing in the data separates this from a coin flip",
}


def _runway_component(r) -> tuple[float, str | None]:
    runway = r.get("runway_at_catalyst")
    if pd.isna(runway):
        return 0.0, None
    if runway < 0:
        return -1.0, f"cash runs out ~{abs(runway):.0f}mo before the catalyst"
    if runway < RUNWAY_CRITICAL_MONTHS:
        return -0.7, f"only {runway:.0f}mo cash left at the catalyst"
    if runway < RUNWAY_WARN_MONTHS:
        return -0.3, f"{runway:.0f}mo cash at the catalyst"
    return 0.2, f"{runway:.0f}mo cash, financing not forced"


def _runup_component(r) -> tuple[float, str | None]:
    ret = r.get("ret_20d")
    if pd.isna(ret):
        return 0.0, None
    if ret > RUNUP_HOT_20D:
        return -0.6, f"+{ret:.0%} in 20 sessions into the date"
    if ret < -0.25:
        return 0.3, f"{ret:.0%} in 20 sessions, expectations reset"
    return 0.0, None


def _vol_component(r) -> tuple[float, str | None]:
    ratio = r.get("move_ratio")
    if pd.isna(ratio):
        return 0.0, None
    if ratio > IMPLIED_RICH_RATIO:
        return 0.0, (f"straddle prices {r['implied_move']:.0%} vs "
                     f"{r['expected_move']:.0%} typical ({ratio:.1f}x)")
    if ratio < 0.8:
        return 0.0, (f"straddle prices only {r['implied_move']:.0%} vs "
                     f"{r['expected_move']:.0%} typical ({ratio:.1f}x)")
    return 0.0, None


def _loa_component(r) -> tuple[float, str | None]:
    loa = r.get("loa")
    if pd.isna(loa):
        return 0.0, None
    score = (loa - 0.4) * 1.2
    note = f"base-rate LOA {loa:.0%} for {r.get('stage')} {r.get('ta')}"
    if r.get("designations"):
        note += f" ({r['designations']})"
    return max(-1.0, min(1.0, score)), note


def classify(r) -> dict:
    """Score one catalyst row."""
    runway_s, runway_why = _runway_component(r)
    runup_s, runup_why = _runup_component(r)
    _, vol_why = _vol_component(r)
    loa_s, loa_why = _loa_component(r)

    reasons = [w for w in (runway_why, runup_why, vol_why, loa_why) if w]
    if pd.isna(r.get("materiality")) and pd.isna(r.get("market_cap")):
        reasons.append("size unknown: no share count filed, materiality untested")
    elif pd.notna(r.get("materiality")) and r["materiality"] < 1.0:
        reasons.append(
            f"immaterial: asset is ~{r['expected_move']:.0%} of cap vs "
            f"{r['baseline_move']:.0%} baseline noise")
    ratio = r.get("move_ratio")
    conf = float(r.get("date_confidence") or 0.0)

    # Setup selection, most actionable first.
    setup = "NO_EDGE"
    lean = "flat"

    # A catalyst that cannot move the stock further than its own noise is not
    # tradeable as an event, whatever the straddle costs.
    materiality = r.get("materiality")
    material = pd.notna(materiality) and materiality >= 1.0

    # Still no hard market-cap gate, but for the opposite reason to the one
    # first recorded here. That earlier note claimed microcaps carry positive
    # mean abnormal returns; it was computed from market caps that mixed
    # split-adjusted prices with pre-split share counts, which inflated 20% of
    # observations (one by 53,000,000x) and flipped the sign. Corrected,
    # microcaps run -10.5% @126d and large caps +8.8%, both surviving multiple
    # testing. Microcap now *raises* conviction on a short instead of vetoing
    # it, and heavy 8-K activity raises it further: low runway + 6+ 8-Ks in 90
    # days + microcap was -39.8% @126d on a 19% hit rate.
    if runway_s <= -0.7:
        setup, lean = "DILUTION_SHORT", "short"
    elif material and pd.notna(ratio) and ratio > IMPLIED_RICH_RATIO:
        setup, lean = "VOL_SELL_RICH", "short vol"
    elif material and pd.notna(ratio) and ratio < 0.8:
        setup, lean = "VOL_BUY_CHEAP", "long vol"
    elif runup_s <= -0.6 and loa_s < 0 and r.get("is_binary"):
        setup, lean = "RUNUP_FADE", "short"
    elif loa_s > 0.25 and runway_s >= 0.2 and (r.get("ret_20d") or 0) < RUNUP_HOT_20D:
        setup, lean = "BASE_RATE_LONG", "long"

    # A crowded short is how this setup goes wrong: the thesis can be correct
    # and still be unholdable if everyone is already on the same side. This
    # only ever *reduces* conviction on a short -- it never creates a long.
    # Calibrated, not guessed. Splitting low-runway names on 5 days to cover:
    # uncrowded ran -7.2% @21d and -20.2% @126d (both surviving multiple
    # testing), crowded ran -1.1% (p=0.63) and -6.4% (p=0.23) -- the edge does
    # not survive a crowded exit. Crowded shorts on well-funded names drifted
    # the other way, +3.1% @63d (p=0.001). So this cuts hard at the measured
    # threshold rather than tapering politely.
    squeeze = r.get("squeeze_risk")
    squeeze_penalty = 1.0
    if lean == "short" and pd.notna(squeeze) and squeeze:
        squeeze_penalty = {1: 0.85, 2: 0.40, 3: 0.25}.get(int(squeeze), 1.0)
        dtc = r.get("days_to_cover")
        reasons.append(
            f"squeeze risk {r.get('squeeze_label')}"
            + (f": {dtc:.1f} days to cover" if pd.notna(dtc) else "")
            + " -- crowded low-runway shorts tested -1.1% @21d (p=0.63) vs "
              "-7.2% uncrowded")
    build = r.get("short_build")
    if lean == "short" and pd.notna(build) and build > 0.5:
        reasons.append(f"short interest +{build:.0%} since last reading")

    # Insider activity is displayed, never scored: it has not been through the
    # harness yet. But a C-suite buy or a cluster of insiders buying while the
    # engine is calling a short is a contradiction worth reading before acting,
    # so it is surfaced as a caution rather than silently ignored.
    if lean == "short":
        if r.get("cluster_buy"):
            reasons.append(
                f"CAUTION: {int(r.get('n_buyers') or 0)} insiders bought "
                "(cluster) -- untested, not scored")
        senior = r.get("senior_net_usd")
        if pd.notna(senior) and senior > 100_000:
            reasons.append(
                f"CAUTION: C-suite net bought ${senior:,.0f} -- untested, not scored")
        desc = r.get("biggest_move_desc")
        big = r.get("biggest_delta_own")
        if desc and pd.notna(big) and big > 0.5:
            reasons.append(f"CAUTION: {desc}")

    # Conviction blends signal strength with how well-dated the catalyst is.
    strength = abs(runway_s) * 0.4 + abs(runup_s) * 0.25 + abs(loa_s) * 0.35

    # Measured amplifiers on the one validated setup.
    if setup == "DILUTION_SHORT":
        # Serial issuance is what the runway signal was actually picking up.
        # Split on two offerings in 24 months: serial issuers ran -8.8% @21d
        # and -27.7% @126d (both survive multiple testing) while rare issuers
        # ran -0.3% (p=0.87) and +2.2% (p=0.64) -- nothing at all. Low cash
        # only predicts a decline when the company habitually raises.
        serial = (r.get("offerings_24m") or 0) >= 2
        if serial:
            strength = min(1.0, strength * 1.35)
            reasons.append(
                f"{int(r['offerings_24m'])} offerings in 24mo: serial issuer "
                "(-27.7% @126d vs +2.2% for rare issuers)")
        else:
            strength *= 0.5
            reasons.append("rare issuer: low runway alone tested flat (p=0.87)")
        if r.get("dilution_label") == "loaded":
            reasons.append("dilution readiness loaded (-17.0% @126d)")
        since = r.get("days_since_offering")
        if pd.notna(since) and since <= 120:
            reasons.append(f"priced a deal {int(since)}d ago (-9.0% @126d)")

        busy = (r.get("eightk_90d") or 0) >= 6
        if r.get("microcap"):
            strength = min(1.0, strength * 1.25)
            reasons.append("microcap: -10.5% abn @126d for the size bucket")
        if busy:
            strength = min(1.0, strength * 1.3)
            reasons.append(f"{int(r['eightk_90d'])} 8-Ks in 90d: heavy filing activity")
        if busy and r.get("microcap"):
            reasons.append("low runway + busy + microcap tested -39.8% @126d")
    if setup in ("VOL_SELL_RICH", "VOL_BUY_CHEAP") and pd.notna(ratio):
        strength = max(strength, min(1.0, abs(ratio - 1.0)))
    if setup == "NO_EDGE":
        strength *= 0.3

    conviction = round(100 * strength * (0.35 + 0.65 * conf) * squeeze_penalty)
    if r.get("illiquid"):
        conviction = round(conviction * 0.6)
        reasons.append("thin: under $1M/day traded")

    evidence, evidence_note = SETUP_EVIDENCE.get(setup, ("untested", ""))
    verdict, directional = VERDICT.get(setup, ("NO TRADE", False))

    return {
        "verdict": verdict,
        "directional": directional,
        "setup": setup,
        "setup_desc": SETUPS[setup],
        "evidence": evidence,
        "evidence_note": evidence_note,
        "lean": lean,
        "conviction": int(min(100, max(0, conviction))),
        "reasons": " | ".join(reasons) if reasons else "insufficient data",
        "runway_score": round(runway_s, 2),
        "runup_score": round(runup_s, 2),
        "loa_score": round(loa_s, 2),
    }


def score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    scored = pd.DataFrame([classify(r) for _, r in df.iterrows()], index=df.index)
    out = pd.concat([df, scored], axis=1)
    return out.sort_values(["conviction", "days_to_catalyst"],
                           ascending=[False, True])
