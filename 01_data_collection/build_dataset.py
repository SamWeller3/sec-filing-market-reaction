"""
Combine the filings metadata + price history into one
master table, and sanity-check that every filing has enough surrounding
price data to actually be usable in the event study.

Usage:
    python build_dataset.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ESTIMATION_WINDOW_TRADING_DAYS = 250  # ~1 trading year, for the market-model regression later on
EVENT_WINDOW_TRADING_DAYS = 5  # days after filing we need price data for


def main():
    filings = pd.read_parquet(config.FILINGS_METADATA_PATH)
    prices = pd.read_parquet(config.PRICES_PATH)

    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    prices["date"] = pd.to_datetime(prices["date"])

    print(f"Loaded {len(filings)} filings and {len(prices)} price rows "
          f"({prices['ticker'].nunique()} tickers)")

    rows = []
    dropped_no_text = 0
    dropped_insufficient_history = 0
    dropped_insufficient_forward = 0

    for ticker, group in filings.groupby("ticker"):
        ticker_prices = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if ticker_prices.empty:
            dropped_insufficient_history += len(group)
            continue

        for _, filing in group.iterrows():
            if not filing.get("text_path"):
                dropped_no_text += 1
                continue

            # how many trading days of price history exist before this filing
            before = ticker_prices[ticker_prices["date"] < filing["filing_date"]]
            after = ticker_prices[ticker_prices["date"] >= filing["filing_date"]]

            if len(before) < ESTIMATION_WINDOW_TRADING_DAYS:
                dropped_insufficient_history += 1
                continue
            if len(after) < EVENT_WINDOW_TRADING_DAYS:
                dropped_insufficient_forward += 1
                continue

            rows.append(
                {
                    "ticker": filing["ticker"],
                    "cik": filing["cik"],
                    "accession_number": filing["accession_number"],
                    "filing_date": filing["filing_date"],
                    "text_path": filing["text_path"],
                    "char_count": filing.get("char_count", 0),
                    "trading_days_before": len(before),
                    "trading_days_after": len(after),
                }
            )

    master = pd.DataFrame(rows)
    master.to_parquet(config.MASTER_DATASET_PATH, index=False)

    print("\nSummary:")
    print(f"Usable filing-events:            {len(master)}")
    print(f"Dropped (no filing text):         {dropped_no_text}")
    print(f"Dropped (not enough history):     {dropped_insufficient_history}")
    print(f"Dropped (not enough forward data):{dropped_insufficient_forward}")
    if len(master):
        print(f"\nDate range: {master['filing_date'].min().date()} -> "
              f"{master['filing_date'].max().date()}")
        print(f"\nFilings per ticker:\n{master['ticker'].value_counts()}")
    print(f"\nSaved to {config.MASTER_DATASET_PATH}")


if __name__ == "__main__":
    main()
