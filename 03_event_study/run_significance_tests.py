"""
The actual hypothesis tests: is there a market reaction to these filings
at all, and does sentiment/materiality predict it?

Run compute_abnormal_returns.py first. Prints effect size, CI, and p-value
together for every test, not just a pass/fail on significance.

Usage:
    python run_significance_tests.py
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_test_result(name, estimate, ci, statistic, p_value, n, adjusted_p=None, statistic_name="statistic"):
    print(f"\n{name}")
    print(f"  n = {n}")
    print(f"  estimate = {estimate:.4f}   95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"  {statistic_name} = {statistic:.4f}   p = {p_value:.4f}"
          + (f"   (BH-adjusted p = {adjusted_p:.4f})" if adjusted_p is not None else ""))


def test_mean_car(df: pd.DataFrame):
    car = df["car_primary"].dropna().values
    n = len(car)
    mean_car = car.mean()
    se_car = car.std(ddof=1) / np.sqrt(n)
    t_stat, p_value = stats.ttest_1samp(car, popmean=0.0)
    ci = (mean_car - 1.96 * se_car, mean_car + 1.96 * se_car)

    print_test_result(
        f"1. Mean CAR over [0,+{config.EVENT_WINDOW_PRIMARY_DAYS}] across all events != 0?",
        mean_car, ci, t_stat, p_value, n, statistic_name="t-stat",
    )
    print(f"  -> mean CAR = {mean_car * 100:.3f}% over the {config.EVENT_WINDOW_PRIMARY_DAYS + 1}-day window")


def test_sentiment_relationship(df: pd.DataFrame):
    n = len(df)
    pearson_r, pearson_p = stats.pearsonr(df["sentiment_score"], df["car_primary"])
    spearman_r, spearman_p = stats.spearmanr(df["sentiment_score"], df["car_primary"])

    print(f"\n2. Sentiment score <-> CAR relationship")
    print(f"  n = {n}")
    print(f"  Pearson r  = {pearson_r:.4f}   p = {pearson_p:.4f}")
    print(f"  Spearman r = {spearman_r:.4f}   p = {spearman_p:.4f}")

    model = smf.ols("car_primary ~ sentiment_score", data=df).fit()
    slope = model.params["sentiment_score"]
    ci_low, ci_high = model.conf_int().loc["sentiment_score"]
    slope_p = model.pvalues["sentiment_score"]

    print(f"  OLS slope  = {slope:.4f}   95% CI = ({ci_low:.4f}, {ci_high:.4f})   "
          f"p = {slope_p:.4f}   R^2 = {model.rsquared:.4f}")
    print(f"  -> a 1-point increase in sentiment_score is associated with a "
          f"{slope * 100:.3f} percentage-point change in CAR")

    return model


def test_materiality_groups(df: pd.DataFrame):
    n_unknown = (df["materiality"] == "unknown").sum()
    df = df[df["materiality"].isin(["high", "medium", "low"])]
    print(f"\n3. CAR across materiality buckets (high/medium/low)")
    print(f"  {n_unknown} filing(s) excluded due to unknown materiality rating; n={len(df)} for this test")

    groups = {level: df.loc[df["materiality"] == level, "car_primary"].values
              for level in ["high", "medium", "low"]}
    for level, values in groups.items():
        print(f"  {level}: n={len(values)}, mean CAR={values.mean() * 100:.3f}%, std={values.std(ddof=1):.4f}")

    f_stat, anova_p = stats.f_oneway(*groups.values())
    h_stat, kw_p = stats.kruskal(*groups.values())
    print(f"  One-way ANOVA (secondary):    F = {f_stat:.4f}   p = {anova_p:.4f}")
    print(f"  Kruskal-Wallis (primary; CAR unlikely to be normal): H = {h_stat:.4f}   p = {kw_p:.4f}")

    if kw_p >= 0.05:
        print("  Omnibus test not significant -> skipping pairwise comparisons.")
        return

    print("  Omnibus test significant -> running pairwise Mann-Whitney U comparisons "
          "with Benjamini-Hochberg FDR correction:")
    pairs = [("high", "medium"), ("high", "low"), ("medium", "low")]
    raw_pvals = []
    for a, b in pairs:
        _, p = stats.mannwhitneyu(groups[a], groups[b])
        raw_pvals.append(p)

    _, adjusted_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")
    for (a, b), raw_p, adj_p in zip(pairs, raw_pvals, adjusted_pvals):
        print(f"    {a} vs {b}: raw p = {raw_p:.4f}   BH-adjusted p = {adj_p:.4f}")


def make_summary_chart(df: pd.DataFrame, model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    materiality_df = df[df["materiality"].isin(["high", "medium", "low"])]
    order = ["low", "medium", "high"]
    data = [materiality_df.loc[materiality_df["materiality"] == level, "car_primary"] for level in order]
    axes[0].boxplot(data, tick_labels=order, showmeans=True)
    for i, d in enumerate(data, start=1):
        jitter = np.random.normal(0, 0.04, size=len(d))
        axes[0].scatter(np.full(len(d), i) + jitter, d, alpha=0.4, s=15, color="steelblue")
    axes[0].set_xlabel("Materiality")
    axes[0].set_ylabel(f"CAR [0,+{config.EVENT_WINDOW_PRIMARY_DAYS}]")
    axes[0].set_title("CAR by materiality bucket")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")

    x = df["sentiment_score"]
    y = df["car_primary"]
    axes[1].scatter(x, y, alpha=0.4, s=15, color="steelblue")
    x_sorted = np.linspace(x.min(), x.max(), 100)
    pred = model.get_prediction(pd.DataFrame({"sentiment_score": x_sorted})).summary_frame(alpha=0.05)
    axes[1].plot(x_sorted, pred["mean"], color="firebrick")
    axes[1].fill_between(x_sorted, pred["mean_ci_lower"], pred["mean_ci_upper"], color="firebrick", alpha=0.15)
    axes[1].set_xlabel("Sentiment score")
    axes[1].set_ylabel(f"CAR [0,+{config.EVENT_WINDOW_PRIMARY_DAYS}]")
    axes[1].set_title("CAR vs. sentiment score")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(config.EVENT_STUDY_CHART_PATH, dpi=150)
    print(f"\nSaved summary chart to {config.EVENT_STUDY_CHART_PATH}")


def main():
    returns = pd.read_parquet(config.ABNORMAL_RETURNS_PATH)
    scores = pd.read_parquet(config.FILING_SCORES_PATH)

    df = returns.merge(scores, on=["ticker", "accession_number"], suffixes=("", "_score"))

    print(f"Merged {len(df)} events with both abnormal returns and filing scores.\n")
    print("=" * 70)

    test_mean_car(df)
    model = test_sentiment_relationship(df)
    test_materiality_groups(df)
    make_summary_chart(df, model)

    results_summary = pd.DataFrame([
        {"test": "mean_car_primary", "n": len(df)},
    ])
    results_summary.to_csv(config.EVENT_STUDY_RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("Notes:")
    print(f"  n={len(df)}, single ~1-year window, 40 tickers. Treat these as a")
    print("  preliminary signal, not a definitive result. A null here for")
    print("  sentiment/materiality is a real possible outcome, not a bug: routine")
    print("  8-Ks may get priced in faster than a few-day window catches, or")
    print("  sentiment_score may not capture what actually moves price for these")
    print("  filing types (earnings, exec changes, debt issuance, etc.).")


if __name__ == "__main__":
    main()
