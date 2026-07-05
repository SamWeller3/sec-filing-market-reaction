"""
Compares the LLM's scores against the hand labels in labeling_sample.csv.
Fill in 'your_sentiment' and 'your_materiality' there first.
"""

import os
import sys

import pandas as pd

# Let this script find config.py at the repo root, regardless of which
# subfolder it's sitting in.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    labeled = pd.read_csv(config.LABELING_SAMPLE_PATH)

    if labeled["your_sentiment"].isna().all() or (labeled["your_sentiment"] == "").all():
        print("No hand labels found yet. Fill in 'your_sentiment' and")
        print("'your_materiality' in data/labeling_sample.csv before running this.")
        return

    scores = pd.read_parquet(config.FILING_SCORES_PATH)
    merged = labeled.merge(scores, on=["ticker", "accession_number"], how="inner", suffixes=("", "_model"))

    merged["your_sentiment"] = pd.to_numeric(merged["your_sentiment"], errors="coerce")
    merged = merged.dropna(subset=["your_sentiment", "sentiment_score"])

    print(f"Comparing {len(merged)} hand-labeled filings against model scores.\n")

    correlation = merged["your_sentiment"].corr(merged["sentiment_score"])
    mae = (merged["your_sentiment"] - merged["sentiment_score"]).abs().mean()
    print(f"Sentiment correlation (yours vs. model):  {correlation:.3f}")
    print(f"Sentiment mean absolute error:             {mae:.3f}")

    merged["your_materiality"] = merged["your_materiality"].str.strip().str.lower()
    merged["materiality"] = merged["materiality"].str.strip().str.lower()
    agreement = (merged["your_materiality"] == merged["materiality"]).mean()
    print(f"Materiality exact-match agreement:         {agreement:.1%}")

    print("\nBiggest disagreements (by sentiment gap), worth reading to understand why:")
    merged["gap"] = (merged["your_sentiment"] - merged["sentiment_score"]).abs()
    worst = merged.sort_values("gap", ascending=False).head(5)
    print(worst[["ticker", "your_sentiment", "sentiment_score", "your_materiality", "materiality", "rationale"]]
          .to_string(index=False))

    print("\nHow to read this:")
    print("Correlation > ~0.6 and materiality agreement > ~70% is a reasonable bar")
    print("to move forward. Below that, look at the disagreements above.")


if __name__ == "__main__":
    main()
