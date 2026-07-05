"""
Fit a market model (stock return ~ SPY return) per event over the trading
days before each filing, then use it to compute abnormal returns and
cumulative abnormal returns (CAR) over the days following the filing.

Window sizes live in config.py (ESTIMATION_WINDOW_TRADING_DAYS,
EVENT_WINDOW_PRIMARY_DAYS, EVENT_WINDOW_ROBUSTNESS_DAYS), locked in before
run_significance_tests.py ever ran, so don't tune them after seeing results.

Usage:
    python compute_abnormal_returns.py
"""

import os
import sys

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MIN_ESTIMATION_OBS = 200  # guard: require most of the requested 250-day window


def fit_market_model(ticker_prices: pd.DataFrame, filing_date: pd.Timestamp):
    """OLS of daily_return ~ market_return over the estimation window
    immediately preceding filing_date. Returns (alpha, beta, n_obs)."""
    window = ticker_prices[ticker_prices["date"] < filing_date].tail(
        config.ESTIMATION_WINDOW_TRADING_DAYS
    )
    window = window.dropna(subset=["daily_return", "market_return"])

    if len(window) < MIN_ESTIMATION_OBS:
        return None, None, len(window)

    X = sm.add_constant(window["market_return"])
    model = sm.OLS(window["daily_return"], X).fit()
    alpha, beta = model.params["const"], model.params["market_return"]
    return alpha, beta, len(window)


def compute_event_window_returns(ticker_prices: pd.DataFrame, filing_date: pd.Timestamp, n_days: int):
    """Trading days [0, n_days] relative to filing_date (day 0 = filing_date
    or the next trading day on/after it, if filing_date itself isn't a
    trading day for this ticker)."""
    on_or_after = ticker_prices[ticker_prices["date"] >= filing_date].sort_values("date")
    return on_or_after.head(n_days + 1)


def compute_car(alpha: float, beta: float, event_window: pd.DataFrame) -> float:
    expected_return = alpha + beta * event_window["market_return"]
    abnormal_return = event_window["daily_return"] - expected_return
    return abnormal_return.sum()


def process_all_events(master: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = 0

    for _, event in master.iterrows():
        ticker_prices = prices[prices["ticker"] == event["ticker"]].sort_values("date")
        filing_date = event["filing_date"]

        alpha, beta, n_obs = fit_market_model(ticker_prices, filing_date)
        if alpha is None:
            print(f"  skipping {event['ticker']} {event['accession_number']}: "
                  f"only {n_obs} non-NaN estimation-window observations (need {MIN_ESTIMATION_OBS})")
            skipped += 1
            continue

        primary_window = compute_event_window_returns(ticker_prices, filing_date, config.EVENT_WINDOW_PRIMARY_DAYS)
        robust_window = compute_event_window_returns(ticker_prices, filing_date, config.EVENT_WINDOW_ROBUSTNESS_DAYS)

        car_primary = compute_car(alpha, beta, primary_window)
        car_robust = compute_car(alpha, beta, robust_window)

        expected_primary = alpha + beta * primary_window["market_return"]
        ar_primary = (primary_window["daily_return"] - expected_primary).reset_index(drop=True)

        rows.append({
            "ticker": event["ticker"],
            "accession_number": event["accession_number"],
            "filing_date": filing_date,
            "alpha": alpha,
            "beta": beta,
            "estimation_n_obs": n_obs,
            "car_primary": car_primary,
            "car_robust": car_robust,
            "ar_day0": ar_primary.iloc[0] if len(ar_primary) > 0 else None,
            "ar_day1": ar_primary.iloc[1] if len(ar_primary) > 1 else None,
            "ar_day2": ar_primary.iloc[2] if len(ar_primary) > 2 else None,
        })

    print(f"\nProcessed {len(rows)} events, skipped {skipped}.")
    return pd.DataFrame(rows)


def main():
    master = pd.read_parquet(config.MASTER_DATASET_PATH)
    prices = pd.read_parquet(config.PRICES_PATH)

    master["filing_date"] = pd.to_datetime(master["filing_date"])
    prices["date"] = pd.to_datetime(prices["date"])

    print(f"Loaded {len(master)} events and {len(prices)} price rows.")

    results = process_all_events(master, prices)
    results.to_parquet(config.ABNORMAL_RETURNS_PATH, index=False)

    print(f"\nSaved {len(results)} rows to {config.ABNORMAL_RETURNS_PATH}")
    print("\nBeta distribution:")
    print(results["beta"].describe())
    print(f"\nCAR primary window [0,+{config.EVENT_WINDOW_PRIMARY_DAYS}] distribution:")
    print(results["car_primary"].describe())
    print(f"\nCAR robustness window [0,+{config.EVENT_WINDOW_ROBUSTNESS_DAYS}] distribution:")
    print(results["car_robust"].describe())


if __name__ == "__main__":
    main()
