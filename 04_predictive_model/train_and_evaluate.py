"""
Train and evaluate models predicting car_primary from filing features,
compared against naive baselines, using strictly time-based validation.

Run this AFTER build_features.py. Two evaluation modes: 5-fold expanding
time-series CV (the number to trust for stability across different time
windows) and a single chronological holdout on the most recent ~20% of
events (the headline "how would this do on genuinely future data" result).
Window/model choices below were fixed before looking at results.

Usage:
    python train_and_evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONTINUOUS_FEATURES = ["beta", "char_count", "sentiment_score"]
CATEGORICAL_FEATURES = ["materiality", "sector"]
BINARY_FLAG_FEATURES = [
    "is_earnings", "is_executive_change", "is_ma", "is_debt",
    "is_shareholder_meeting", "is_litigation", "is_capital_return", "is_guidance",
]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + BINARY_FLAG_FEATURES


def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ("continuous", StandardScaler(), CONTINUOUS_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("binary", "passthrough", BINARY_FLAG_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def get_models():
    return {
        "Ridge": build_pipeline(Ridge(alpha=1.0, random_state=config.RANDOM_SEED)),
        "RandomForest": build_pipeline(RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=10,
            random_state=config.RANDOM_SEED,
        )),
        "Baseline (zero)": build_pipeline(DummyRegressor(strategy="constant", constant=0.0)),
        "Baseline (mean)": build_pipeline(DummyRegressor(strategy="mean")),
    }


def directional_accuracy(y_true, y_pred, is_zero_baseline=False):
    if is_zero_baseline:
        # A constant-zero prediction has no real sign, so report the base
        # rate of the majority actual sign instead of a meaningless number.
        return max((y_true > 0).mean(), (y_true <= 0).mean())
    return (np.sign(y_pred) == np.sign(y_true)).mean()


def evaluate(y_true, y_pred, is_zero_baseline=False):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    dir_acc = directional_accuracy(y_true, y_pred, is_zero_baseline)
    return rmse, mae, dir_acc


def run_cv(df, model_name, pipeline):
    tscv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS)
    rmses, maes, dir_accs = [], [], []
    is_zero_baseline = model_name == "Baseline (zero)"

    for train_idx, test_idx in tscv.split(df):
        train_fold = df.iloc[train_idx]
        test_fold = df.iloc[test_idx]

        pipeline.fit(train_fold[ALL_FEATURES], train_fold["car_primary"])
        preds = pipeline.predict(test_fold[ALL_FEATURES])

        rmse, mae, dir_acc = evaluate(test_fold["car_primary"].values, preds, is_zero_baseline)
        rmses.append(rmse)
        maes.append(mae)
        dir_accs.append(dir_acc)

    return {
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "dir_acc_mean": np.mean(dir_accs), "dir_acc_std": np.std(dir_accs),
    }


def run_holdout(df, model_name, pipeline, cutoff_idx):
    train = df.iloc[:cutoff_idx]
    test = df.iloc[cutoff_idx:]
    is_zero_baseline = model_name == "Baseline (zero)"

    pipeline.fit(train[ALL_FEATURES], train["car_primary"])
    preds = pipeline.predict(test[ALL_FEATURES])

    rmse, mae, dir_acc = evaluate(test["car_primary"].values, preds, is_zero_baseline)
    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc}, test


def print_model_comparison(name, cv, holdout, baseline_holdout):
    print(f"\n{name}")
    print(f"  5-fold CV:      RMSE = {cv['rmse_mean']*100:.3f}% +/- {cv['rmse_std']*100:.3f}%   "
          f"MAE = {cv['mae_mean']*100:.3f}% +/- {cv['mae_std']*100:.3f}%   "
          f"directional accuracy = {cv['dir_acc_mean']*100:.1f}% +/- {cv['dir_acc_std']*100:.1f}%")
    print(f"  Final holdout:  RMSE = {holdout['rmse']*100:.3f}%   "
          f"MAE = {holdout['mae']*100:.3f}%   "
          f"directional accuracy = {holdout['dir_acc']*100:.1f}%")

    beats_baseline = holdout["rmse"] < baseline_holdout["rmse"]
    print(f"  vs. zero baseline (holdout RMSE {baseline_holdout['rmse']*100:.3f}%, "
          f"directional accuracy {baseline_holdout['dir_acc']*100:.1f}%): "
          + ("beats" if beats_baseline else "does NOT beat")
          + " the naive baseline on holdout RMSE.")


def make_comparison_chart(results):
    names = list(results.keys())
    holdout_rmse = [results[n]["holdout"]["rmse"] * 100 for n in names]
    holdout_dir_acc = [results[n]["holdout"]["dir_acc"] * 100 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, holdout_rmse, color="steelblue")
    axes[0].set_ylabel("Holdout RMSE (%)")
    axes[0].set_title("Holdout RMSE by model")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, holdout_dir_acc, color="steelblue")
    axes[1].axhline(50, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Holdout directional accuracy (%)")
    axes[1].set_title("Holdout directional accuracy by model")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(config.MODEL_COMPARISON_CHART_PATH, dpi=150)
    print(f"\nSaved comparison chart to {config.MODEL_COMPARISON_CHART_PATH}")


def main():
    df = pd.read_parquet(config.MODEL_FEATURES_PATH)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values("filing_date").reset_index(drop=True)

    cutoff_idx = int(len(df) * (1 - config.HOLDOUT_FRACTION))
    print(f"n={len(df)}, holdout cutoff at row {cutoff_idx} "
          f"(filing_date {df['filing_date'].iloc[cutoff_idx].date()}), "
          f"{len(df) - cutoff_idx} rows in the holdout.")

    holdout_df = df.iloc[cutoff_idx:]
    print("\nHoldout ticker counts:")
    print(holdout_df["ticker"].value_counts())
    print("\nHoldout sector counts:")
    print(holdout_df["sector"].value_counts())
    missing_tickers = set(df["ticker"]) - set(holdout_df["ticker"])
    if missing_tickers:
        print(f"\nTickers absent from the holdout (their most recent filings predate "
              f"the cutoff): {sorted(missing_tickers)}")

    models = get_models()
    results = {}

    print("\n" + "=" * 70)
    for name, pipeline in models.items():
        cv = run_cv(df, name, pipeline)
        holdout, _ = run_holdout(df, name, pipeline, cutoff_idx)
        results[name] = {"cv": cv, "holdout": holdout}

    baseline_holdout = results["Baseline (zero)"]["holdout"]
    for name in models:
        print_model_comparison(name, results[name]["cv"], results[name]["holdout"], baseline_holdout)

    make_comparison_chart(results)

    print("\n" + "=" * 70)
    print("Notes:")
    real_models = ["Ridge", "RandomForest"]
    winners = [m for m in real_models if results[m]["holdout"]["rmse"] < baseline_holdout["rmse"]]
    if winners:
        print(f"  {', '.join(winners)} beat the zero baseline on holdout RMSE. In contrast to")
        print("  the event study's null finding for sentiment/materiality alone, a richer")
        print("  feature set produced a model that outperformed a naive baseline out of")
        print("  sample here. n=527 over a single ~1-year window is still a modest sample,")
        print("  so treat this as a preliminary result, not a settled one.")
    else:
        print("  Neither Ridge nor RandomForest beat the zero baseline on holdout RMSE.")
        print("  Consistent with the event study's null finding for sentiment/materiality")
        print("  alone, a richer feature set did not produce a model that reliably beats")
        print("  a naive baseline out of sample. That's a legitimate result: it's evidence")
        print("  these features don't carry exploitable information about reaction size,")
        print("  not a failure of the modeling itself.")


if __name__ == "__main__":
    main()
