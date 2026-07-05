"""
Score every filing in the master dataset for sentiment and materiality
using an LLM.

Requires an Anthropic API key in a local .env file:
    ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import os
import sys
import time

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

# Let this script find config.py at the repo root, regardless of which
# subfolder it's sitting in.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SCORING_PROMPT = """You are analyzing an SEC 8-K filing to help predict how the stock market will react to it.

Read the filing text below and respond with ONLY a JSON object (no other text) in this exact format:
{{
  "sentiment_score": <float from -1.0 (very negative) to 1.0 (very positive), 0.0 = neutral/no clear tone>,
  "materiality": "<one of: high, medium, low (how likely this is to move the stock price)>",
  "item_type_summary": "<one short phrase describing what kind of event this is, e.g. 'earnings release', 'executive appointment', 'M&A announcement', 'debt offering'>",
  "rationale": "<one sentence explaining your sentiment_score and materiality rating>"
}}

Filing text:
{filing_text}
"""


def score_one_filing(filing_text: str) -> dict:
    truncated = filing_text[: config.MAX_FILING_CHARS]

    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": SCORING_PROMPT.format(filing_text=truncated)}],
    )
    raw_text = response.content[0].text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    return json.loads(raw_text)


def main():
    master = pd.read_parquet(config.MASTER_DATASET_PATH)

    if os.path.exists(config.FILING_SCORES_PATH):
        existing = pd.read_parquet(config.FILING_SCORES_PATH)
        done_accessions = set(existing["accession_number"])
        print(f"Resuming: {len(done_accessions)} filings already scored.")
    else:
        existing = pd.DataFrame()
        done_accessions = set()

    results = []
    failures = 0

    for i, row in master.iterrows():
        if row["accession_number"] in done_accessions:
            continue

        try:
            with open(row["text_path"], "r", encoding="utf-8") as f:
                filing_text = f.read()

            score = score_one_filing(filing_text)
            results.append({
                "ticker": row["ticker"],
                "accession_number": row["accession_number"],
                "filing_date": row["filing_date"],
                "sentiment_score": score.get("sentiment_score"),
                "materiality": score.get("materiality"),
                "item_type_summary": score.get("item_type_summary"),
                "rationale": score.get("rationale"),
            })
        except Exception as e:
            print(f"  failed on {row['ticker']} {row['accession_number']}: {e}")
            failures += 1

        if len(results) % 25 == 0 and len(results) > 0:
            print(f"  ...{len(results)} filings scored ({failures} failures so far)")
            combined = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            combined.to_parquet(config.FILING_SCORES_PATH, index=False)

        time.sleep(0.1)

    combined = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    combined.to_parquet(config.FILING_SCORES_PATH, index=False)

    print(f"\nScored {len(results)} filings this run ({failures} failures).")
    print(f"Total scored: {len(combined)} / {len(master)}")
    print(f"Saved to {config.FILING_SCORES_PATH}")
    print("\nSentiment score distribution:")
    print(combined["sentiment_score"].describe())
    print("\nMateriality breakdown:")
    print(combined["materiality"].value_counts())


if __name__ == "__main__":
    main()
