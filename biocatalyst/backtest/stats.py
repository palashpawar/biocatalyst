"""Inference for the event study.

Two choices matter here and both widen the error bars rather than narrow them:

1. **Cluster by ticker.** One company contributes many events, and its events
   are correlated with each other. Treating them as independent observations
   inflates n and produces confidence intervals that look far better than the
   evidence supports.
2. **Bootstrap rather than t-tests.** Event returns are heavily fat-tailed --
   a biotech readout produces -80% and +300% days -- so the normal
   approximation behind a t-statistic does not hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CLUSTERS = 12   # below this, a bucket is reported but never called a result


def cluster_bootstrap(df: pd.DataFrame, value_col: str,
                      cluster_col: str = "ticker", n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Mean with a ticker-clustered bootstrap confidence interval."""
    sub = df[[value_col, cluster_col]].dropna()
    if sub.empty:
        return {"n": 0, "n_clusters": 0, "mean": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "p_two_sided": np.nan}

    groups = [g[value_col].to_numpy() for _, g in sub.groupby(cluster_col)]
    n_clusters = len(groups)
    observed = float(sub[value_col].mean())

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        vals = np.concatenate([groups[i] for i in pick])
        means[b] = vals.mean()

    lo, hi = np.percentile(means, [2.5, 97.5])
    # Bootstrap p-value: how often a recentred resample crosses zero.
    centred = means - means.mean()
    p = float(np.mean(np.abs(centred) >= abs(observed)))

    return {
        "n": int(len(sub)),
        "n_clusters": int(n_clusters),
        "mean": observed,
        "median": float(sub[value_col].median()),
        "hit_rate": float((sub[value_col] > 0).mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_two_sided": p,
    }


def summarize(df: pd.DataFrame, by: str | None, horizons=(1, 5, 21, 63),
              n_boot: int = 2000) -> pd.DataFrame:
    """Per-bucket abnormal-return summary across horizons."""
    if df.empty:
        return pd.DataFrame()
    groups = [("ALL", df)] if by is None else list(df.groupby(by))
    rows = []
    for name, g in groups:
        for h in horizons:
            col = f"abn_{h}d"
            if col not in g.columns:
                continue
            s = cluster_bootstrap(g, col, n_boot=n_boot)
            rows.append({"bucket": name, "horizon": f"{h}d", **s})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["reliable"] = out["n_clusters"] >= MIN_CLUSTERS
    # Nominal significance requires BOTH the interval to exclude zero and the
    # bootstrap p-value to clear 5%. A percentile interval can graze zero on a
    # skewed resampling distribution while p sits near 0.06, and reporting
    # that as a finding is how a backtest talks itself into an edge.
    out["significant"] = (out["reliable"]
                          & ((out["ci_lo"] > 0) | (out["ci_hi"] < 0))
                          & (out["p_two_sided"] < 0.05))
    out["survives_bonferroni"] = False
    return out


def bonferroni_note(n_tests: int, alpha: float = 0.05) -> str:
    """The multiple-comparison threshold this study should be judged against."""
    if n_tests <= 0:
        return ""
    return (f"{n_tests} buckets tested; a p-value below "
            f"{alpha / n_tests:.4f} is the Bonferroni-corrected bar for "
            f"{alpha:.0%} family-wise error.")


def apply_multiple_testing(tables: list[pd.DataFrame], alpha: float = 0.05) -> int:
    """Mark which rows clear the family-wise bar across every bucket tested.

    Each table is modified in place. Returns the number of tests in the family.
    """
    live = [t for t in tables if t is not None and not t.empty]
    n_tests = sum(len(t) for t in live)
    if not n_tests:
        return 0
    bar = alpha / n_tests
    for t in live:
        t["survives_bonferroni"] = t["reliable"] & (t["p_two_sided"] < bar)
    return n_tests


def format_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "(no results)"
    lines = [f"{'bucket':22}{'hz':>5}{'n':>6}{'tick':>6}"
             f"{'mean':>9}{'median':>9}{'hit':>7}{'95% CI':>20}{'p':>8}  flag"]
    lines.append("-" * 100)
    for _, r in summary.iterrows():
        flag = ""
        if not r["reliable"]:
            flag = f"thin (n_tickers={r['n_clusters']})"
        elif r.get("survives_bonferroni"):
            flag = "*** survives multiple testing"
        elif r["significant"]:
            flag = "*  nominal only, fails multiple testing"
        ci = f"[{r['ci_lo']:+.1%}, {r['ci_hi']:+.1%}]"
        lines.append(
            f"{str(r['bucket'])[:21]:22}{r['horizon']:>5}{r['n']:>6}"
            f"{r['n_clusters']:>6}{r['mean']:>+9.2%}{r['median']:>+9.2%}"
            f"{r['hit_rate']:>7.0%}{ci:>20}{r['p_two_sided']:>8.3f}  {flag}")
    return "\n".join(lines)
