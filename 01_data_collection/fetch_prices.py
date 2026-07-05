"""
Pull daily historical price data for every ticker in the basket, covering
enough history before/after each filing to later run the market-model
event study.

Uses yfinance for now since this is a batch/historical pull, not the live
feed, and it needs no signup. Swap this out for Alpaca's historical endpoint
later if you want the whole project on one data vendor for consistency.

Usage:
    python fetch_prices.py
"""

import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Extra padding around the 8-K lookback window:
#   - ~13 months back gives a ~250-trading-day estimation window before the
#     earliest filing (needed for the market-model regression later on).
#   - 15 days forward covers the reaction window after the most recent filing.
PRICE_START_PADDING_DAYS = config.LOOKBACK_DAYS + 380
PRICE_END_PADDING_DAYS = 15


def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    period_start = pd.Timestamp.now() - pd.Timedelta(days=PRICE_START_PADDING_DAYS)
    period_end = pd.Timestamp.now() + pd.Timedelta(days=PRICE_END_PADDING_DAYS)

    hist = yf.Ticker(ticker).history(
        start=period_start.strftime("%Y-%m-%d"),
        end=period_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )
    if hist.empty:
        return pd.DataFrame()

    hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    hist.columns = ["date", "open", "high", "low", "close", "volume"]
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    hist["ticker"] = ticker
    hist["daily_return"] = hist["close"].pct_change()
    return hist


def fetch_market_benchmark() -> pd.DataFrame:
    """SPY as the market proxy for the market-model regression."""
    hist = fetch_ticker_history("SPY")
    hist = hist.rename(columns={"daily_return": "market_return"})
    return hist[["date", "market_return"]]


def main():
    all_prices = []

    print("Fetching market benchmark (SPY)...")
    market = fetch_market_benchmark()

    for ticker in config.TICKERS:
        print(f"[{ticker}] fetching price history...")
        try:
            hist = fetch_ticker_history(ticker)
            if hist.empty:
                print(f"  -> no data returned for {ticker}")
                continue
            hist = hist.merge(market, on="date", how="left")
            # NOTE: this is a naive excess return (stock return minus market
            # return), just for a quick sanity check here. The real abnormal
            # return (estimating each stock's beta via regression over a
            # pre-event window, then computing actual minus expected return)
            # gets computed properly per-event later on. Don't use this
            # column as your event-study result.
            hist["excess_return_naive"] = hist["daily_return"] - hist["market_return"]
            all_prices.append(hist)
            print(f"  -> {len(hist)} trading days")
        except Exception as e:
            print(f"  failed on {ticker}: {e}")
        time.sleep(0.5)  # be polite to Yahoo's endpoint too

    combined = pd.concat(all_prices, ignore_index=True)
    combined.to_parquet(config.PRICES_PATH, index=False)
    print(f"\nSaved {len(combined)} price rows across {combined['ticker'].nunique()} tickers "
          f"to {config.PRICES_PATH}")


if __name__ == "__main__":
    main()
