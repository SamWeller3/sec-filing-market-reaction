"""
Polls SEC EDGAR for new 8-Ks across the ticker basket, scores each new
one, and publishes it to the filings topic. This is the live counterpart
to replay_producer.py.

Usage:
    python filing_producer.py
"""

import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
from confluent_kafka import Producer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "01_data_collection"))
sys.path.insert(0, os.path.join(REPO_ROOT, "02_nlp_scoring"))
import config
from fetch_filings import download_filing_text, get_recent_8ks, get_ticker_to_cik_map
from score_filings import score_one_filing


def load_seen_accessions() -> set:
    master = pd.read_parquet(config.MASTER_DATASET_PATH)
    return set(master["accession_number"])


def main():
    seen = load_seen_accessions()
    print(f"Seeded {len(seen)} already-known accession numbers.")

    ticker_to_cik = get_ticker_to_cik_map()
    time.sleep(config.SEC_REQUEST_DELAY_SECONDS)

    # Truncated to midnight so today's filings (parsed as midnight
    # timestamps) aren't excluded by a same-day comparison against the
    # current time of day.
    cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    producer = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS})
    print(f"Polling {len(config.TICKERS)} tickers every "
          f"{config.FILING_POLL_INTERVAL_SECONDS}s. Ctrl+C to stop.")

    try:
        while True:
            for ticker in config.TICKERS:
                cik = ticker_to_cik.get(ticker)
                if cik is None:
                    continue

                filings = get_recent_8ks(cik, ticker, cutoff_date)
                time.sleep(config.SEC_REQUEST_DELAY_SECONDS)

                for filing in filings:
                    if filing["accession_number"] in seen:
                        continue
                    seen.add(filing["accession_number"])

                    try:
                        text = download_filing_text(cik, filing["accession_number"], filing["primary_document"])
                        score = score_one_filing(text)
                    except Exception as e:
                        print(f"  [{ticker}] failed to process {filing['accession_number']}: {e}")
                        continue

                    message = {
                        "ticker": ticker,
                        "accession_number": filing["accession_number"],
                        "filing_date": filing["filing_date"],
                        "sentiment_score": score.get("sentiment_score"),
                        "materiality": score.get("materiality"),
                        "item_type_summary": score.get("item_type_summary"),
                        "char_count": len(text[: config.MAX_FILING_CHARS]),
                    }
                    producer.produce(config.KAFKA_FILINGS_TOPIC, key=ticker, value=json.dumps(message))
                    producer.poll(0)
                    print(f"  [{ticker}] new filing {filing['accession_number']}: "
                          f"sentiment={message['sentiment_score']}, materiality={message['materiality']}")

            producer.flush()
            time.sleep(config.FILING_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
        producer.flush()


if __name__ == "__main__":
    main()
