"""
Merge the master dataset, filing scores, and abnormal returns into one
feature table for the predictive model.

item_type_summary has 350 distinct freeform values on 527 rows, far too
sparse to one-hot encode without massive overfitting, and dropping it
would throw away a real driver of reaction size. Instead it gets reduced
to 8 keyword-matched binary flags (earnings, executive change, M&A, debt,
shareholder meeting, litigation, capital return, guidance).

Usage:
    python build_features.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ITEM_TYPE_KEYWORDS = {
    "is_earnings": r"earnings|quarterly|q1|q2|q3|q4|results of operations|revenue and",
    "is_executive_change": r"ceo|cfo|coo|cco|executive|director appoint|director resign|"
                            r"director retire|director departure|principal accounting officer|"
                            r"controller|general counsel|succession|leadership transition",
    "is_ma": r"m&a|acquisition|merger|divestiture|disposition",
    "is_debt": r"debt offering|debt issuance|debt facility|debt exchange|credit facility|"
               r"subordinated debt|notes offering|floating rate notes",
    "is_shareholder_meeting": r"shareholder meeting|stockholder meeting|proxy|voting results|annual meeting",
    "is_litigation": r"litigation|lawsuit|settlement|doj|antitrust|investigation",
    "is_capital_return": r"dividend|share repurchase|buyback|stock split",
    "is_guidance": r"guidance",
}


def add_item_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    summary_lower = df["item_type_summary"].str.lower()
    for flag_name, pattern in ITEM_TYPE_KEYWORDS.items():
        df[flag_name] = summary_lower.str.contains(pattern, regex=True, na=False)
    return df


def main():
    master = pd.read_parquet(config.MASTER_DATASET_PATH)
    scores = pd.read_parquet(config.FILING_SCORES_PATH)
    returns = pd.read_parquet(config.ABNORMAL_RETURNS_PATH)

    df = master.merge(scores, on=["ticker", "accession_number"], suffixes=("", "_score"))
    df = df.merge(returns, on=["ticker", "accession_number"], suffixes=("", "_returns"))
    print(f"Merged {len(df)} rows from master_dataset, filing_scores, and abnormal_returns.")

    n_unknown = (df["materiality"] == "unknown").sum()
    df = df[df["materiality"].isin(["high", "medium", "low"])].copy()
    print(f"Dropped {n_unknown} unknown-materiality row(s); n={len(df)} going forward.")

    missing_tickers = set(df["ticker"]) - set(config.TICKER_SECTOR_MAP)
    assert not missing_tickers, f"ticker(s) missing from TICKER_SECTOR_MAP: {missing_tickers}"
    df["sector"] = df["ticker"].map(config.TICKER_SECTOR_MAP)

    df = add_item_type_flags(df)

    keep_cols = [
        "ticker", "accession_number", "filing_date", "sector",
        "sentiment_score", "materiality", "beta", "char_count",
        *ITEM_TYPE_KEYWORDS.keys(),
        "car_primary",
    ]
    df = df[keep_cols]

    df.to_parquet(config.MODEL_FEATURES_PATH, index=False)
    print(f"Saved {len(df)} rows to {config.MODEL_FEATURES_PATH}")

    print("\nSector counts:")
    print(df["sector"].value_counts())

    print("\nItem-type flag coverage:")
    flag_cols = list(ITEM_TYPE_KEYWORDS.keys())
    for flag in flag_cols:
        print(f"  {flag}: {df[flag].sum()}")
    any_flag = df[flag_cols].any(axis=1).sum()
    print(f"  at least one flag: {any_flag} / {len(df)}")


if __name__ == "__main__":
    main()
