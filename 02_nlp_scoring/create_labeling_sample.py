"""
Pulls a random sample of filings to label by hand. validate_scoring.py
compares those labels against the LLM's scores, so we're not just trusting
the LLM because its output sounds plausible.
"""

import os
import sys

import pandas as pd

# Let this script find config.py at the repo root, regardless of which
# subfolder it's sitting in.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    master = pd.read_parquet(config.MASTER_DATASET_PATH)
    sample = master.sample(n=config.LABELING_SAMPLE_SIZE, random_state=config.LABELING_RANDOM_SEED)

    rows = []
    for _, row in sample.iterrows():
        with open(row["text_path"], "r", encoding="utf-8") as f:
            filing_text = f.read()[: config.MAX_FILING_CHARS]

        rows.append({
            "ticker": row["ticker"],
            "accession_number": row["accession_number"],
            "filing_date": row["filing_date"],
            "filing_text": filing_text,
            "your_sentiment": "",
            "your_materiality": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.LABELING_SAMPLE_PATH, index=False)
    print(f"Saved {len(df)} filings to {config.LABELING_SAMPLE_PATH}")
    print("\nOpen it in Excel and fill in 'your_sentiment' (-1.0 to 1.0) and")
    print("'your_materiality' (high/medium/low) for each row by actually reading")
    print("the filing_text column.")


if __name__ == "__main__":
    main()
