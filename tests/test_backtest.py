import datetime as dt
import sys, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.backtest import engine, pit, stats

FACTS = {
    "us-gaap": {
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
            {"end": "2023-03-31", "val": 100, "filed": "2023-05-05", "form": "10-Q"},
            {"end": "2023-06-30", "val": 80, "filed": "2023-08-05", "form": "10-Q"},
            {"end": "2023-09-30", "val": 60, "filed": "2023-11-05", "form": "10-Q"},
            # A restatement of an old period, filed much later.
            {"end": "2023-03-31", "val": 95, "filed": "2024-03-01", "form": "10-K/A"},
        ]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            {"start": "2023-01-01", "end": "2023-03-31", "val": -20,
             "filed": "2023-05-05", "form": "10-Q"},
            {"start": "2023-04-01", "end": "2023-06-30", "val": -20,
             "filed": "2023-08-05", "form": "10-Q"},
        ]}},
    }
}


def test_pit_never_sees_unfiled_periods():
    r = pit.pit_financials("X", dt.date(2023, 9, 1), facts=FACTS)
    # Q3 ends 2023-09-30 but is not filed until November.
    assert r["cash_period_end"] == "2023-06-30"
    assert r["cash_usd"] == 80


def test_pit_excludes_later_restatements():
    # On 2023-06-01 only the original Q1 figure existed, not the 2024 revision.
    r = pit.pit_financials("X", dt.date(2023, 6, 1), facts=FACTS)
    assert r["cash_usd"] == 100


def test_pit_returns_none_before_any_filing():
    assert pit.pit_financials("X", dt.date(2023, 1, 1), facts=FACTS) is None


def test_runway_uses_visible_burn_only():
    r = pit.pit_financials("X", dt.date(2023, 9, 1), facts=FACTS)
    # 80 of cash against 20/quarter -> 12 months.
    assert r["quarterly_burn_usd"] == 20
    assert r["runway_months"] == 12.0


def test_filing_dates_are_within_window():
    ds = pit.filing_dates("X", dt.date(2023, 6, 1), dt.date(2023, 12, 31),
                          facts=FACTS)
    assert ds == [dt.date(2023, 8, 5), dt.date(2023, 11, 5)]


def _panel():
    idx = pd.bdate_range("2023-01-02", periods=60)
    # Stock doubles linearly; benchmark is flat.
    return pd.DataFrame({"AAA": np.linspace(10, 20, 60),
                         engine.BENCHMARK: np.full(60, 100.0)}, index=idx)


def test_abnormal_return_strips_the_benchmark():
    closes = _panel()
    ev = pd.DataFrame([{"ticker": "AAA", "event_date": closes.index[20].date(),
                        "anchor": "test"}])
    r = engine.event_returns(ev, closes, horizons=(5,), entry_offset=5)
    entry, exit_ = closes["AAA"].iloc[15], closes["AAA"].iloc[25]
    assert abs(r["abn_5d"].iloc[0] - (exit_ / entry - 1)) < 1e-9


def test_benchmark_move_is_removed():
    closes = _panel()
    closes[engine.BENCHMARK] = np.linspace(100, 200, 60)  # benchmark now doubles too
    ev = pd.DataFrame([{"ticker": "AAA", "event_date": closes.index[20].date(),
                        "anchor": "test"}])
    r = engine.event_returns(ev, closes, horizons=(5,), entry_offset=5)
    assert abs(r["abn_5d"].iloc[0]) < 0.02   # both rose: little abnormal move


def test_clustering_widens_intervals():
    rng = np.random.default_rng(1)
    # One draw per ticker, repeated 10x: only 20 independent observations.
    vals = np.repeat(rng.normal(0.05, 0.2, 20), 10)
    df = pd.DataFrame({"ticker": np.repeat([f"T{i}" for i in range(20)], 10),
                       "abn_5d": vals})
    clustered = stats.cluster_bootstrap(df, "abn_5d", "ticker")
    naive = stats.cluster_bootstrap(
        df.assign(row=range(len(df))), "abn_5d", "row")
    assert (clustered["ci_hi"] - clustered["ci_lo"]) > \
           (naive["ci_hi"] - naive["ci_lo"]) * 2


def test_thin_buckets_are_never_significant():
    df = pd.DataFrame({"ticker": [f"T{i}" for i in range(4)],
                       "abn_5d": [0.5, 0.52, 0.48, 0.51]})
    out = stats.summarize(df, None, horizons=(5,))
    assert not out["reliable"].iloc[0]
    assert not out["significant"].iloc[0]


def test_bonferroni_marks_only_strong_results():
    df = pd.DataFrame({
        "bucket": ["a", "b"], "horizon": ["5d", "5d"], "n": [200, 200],
        "n_clusters": [40, 40], "mean": [0.1, 0.01],
        "median": [0.1, 0.01], "hit_rate": [0.6, 0.5],
        "ci_lo": [0.05, 0.001], "ci_hi": [0.15, 0.02],
        "p_two_sided": [0.0001, 0.04], "reliable": [True, True],
        "significant": [True, True], "survives_bonferroni": [False, False]})
    n = stats.apply_multiple_testing([df])
    assert n == 2
    assert bool(df["survives_bonferroni"].iloc[0])      # p=0.0001 < 0.025
    assert not bool(df["survives_bonferroni"].iloc[1])  # p=0.04 does not


def test_cadence_is_point_in_time():
    dates = [dt.date(2023, 1, 10), dt.date(2023, 3, 1), dt.date(2023, 6, 15),
             dt.date(2023, 9, 20)]
    c = pit.cadence_at(dates, dt.date(2023, 3, 31))
    # The June and September filings had not happened yet.
    assert c["eightk_90d"] == 2
    assert c["eightk_365d"] == 2
    assert c["days_since_8k"] == 30


def test_cadence_with_no_prior_filings():
    c = pit.cadence_at([dt.date(2024, 1, 1)], dt.date(2023, 1, 1))
    assert c["eightk_90d"] == 0 and c["days_since_8k"] is None


def test_burst_is_relative_to_the_company_baseline():
    # 8 filings a year, 6 of them in the last quarter -> busy for itself.
    base = [dt.date(2023, 1, 1) + dt.timedelta(days=45 * i) for i in range(2)]
    recent = [dt.date(2023, 11, 1) + dt.timedelta(days=5 * i) for i in range(6)]
    c = pit.cadence_at(base + recent, dt.date(2023, 12, 31))
    assert c["burst"] > 2.0


def test_burst_is_none_without_enough_history():
    # One filing all year cannot support a ratio.
    c = pit.cadence_at([dt.date(2023, 12, 1)], dt.date(2023, 12, 31))
    assert c["burst"] is None


def test_cadence_buckets_are_ordered():
    from biocatalyst.backtest.run import _burst_bucket, _count_bucket
    assert _burst_bucket(0.5) < _burst_bucket(1.0) < _burst_bucket(2.0) < _burst_bucket(5.0)
    assert _count_bucket(0) < _count_bucket(1) < _count_bucket(4) < _count_bucket(9)


def test_shares_restated_onto_current_split_basis(monkeypatch):
    # Regression: price history is back-adjusted for splits but EDGAR share
    # counts are not, so raw multiplication inflated market caps by the
    # cumulative split factor -- putting nano-caps in the trillions.
    monkeypatch.setattr(pit, "split_history",
                        lambda t: [(dt.date(2025, 1, 1), 0.1)])
    before = pit.shares_in_current_terms(100e6, "X", dt.date(2024, 6, 30))
    after = pit.shares_in_current_terms(10e6, "X", dt.date(2025, 6, 30))
    assert before == 10e6          # 1-for-10 reverse split applied
    assert after == 10e6           # split already reflected, untouched


def test_no_splits_leaves_shares_unchanged(monkeypatch):
    monkeypatch.setattr(pit, "split_history", lambda t: [])
    assert pit.shares_in_current_terms(5e6, "X", dt.date(2023, 1, 1)) == 5e6


def test_missing_share_count_stays_missing(monkeypatch):
    monkeypatch.setattr(pit, "split_history", lambda t: [])
    assert pit.shares_in_current_terms(None, "X", dt.date(2023, 1, 1)) is None
