"""
Replays historical filings and prices onto the same Kafka topics the live
producers use, on a simulated clock. SEC filings and live trading data
only flow a few hours a day on weekdays, and most days produce zero new
8-Ks across a 40-ticker basket, so a live-only demo would often just show
an empty dashboard. This lets the rest of the pipeline (the reaction
processor, the dashboard) run and be screenshotted on request, using data
already collected.

Delay between events is scaled down from real calendar time by
config.REPLAY_SPEEDUP_FACTOR, capped per step so a long quiet stretch
(a weekend, a gap between filings) doesn't stall the demo.

Usage:
    python replay_producer.py
"""

import json
import os
import sys
import time

import pandas as pd
from confluent_kafka import Producer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

REPLAY_MAX_STEP_DELAY_SECONDS = 2.0


def load_filing_events() -> list[dict]:
    master = pd.read_parquet(config.MASTER_DATASET_PATH)
    scores = pd.read_parquet(config.FILING_SCORES_PATH)
    df = master.merge(scores, on=["ticker", "accession_number"], suffixes=("", "_score"))
    df = df[df["materiality"].isin(["high", "medium", "low"])]

    events = []
    for _, row in df.iterrows():
        events.append({
            "timestamp": row["filing_date"],
            "topic": config.KAFKA_FILINGS_TOPIC,
            "key": row["ticker"],
            "value": {
                "ticker": row["ticker"],
                "accession_number": row["accession_number"],
                "filing_date": row["filing_date"].isoformat(),
                "sentiment_score": row["sentiment_score"],
                "materiality": row["materiality"],
                "item_type_summary": row["item_type_summary"],
                "char_count": int(row["char_count"]),
            },
        })
    return events


def load_price_events(min_date) -> list[dict]:
    # Price history goes back further than the earliest filing, to cover
    # the pre-event estimation window used in the historical market-model
    # regression. That regression already happened offline (its result is
    # the static alpha/beta reaction_processor.py loads), so there's
    # nothing useful for the live pipeline to do with prices from before
    # any filing existed -- replaying them would just be dead time.
    prices = pd.read_parquet(config.PRICES_PATH)
    prices = prices.dropna(subset=["daily_return", "market_return"])
    prices = prices[prices["date"] >= min_date]

    events = []
    for _, row in prices.iterrows():
        events.append({
            "timestamp": row["date"],
            "topic": config.KAFKA_PRICES_TOPIC,
            "key": row["ticker"],
            "value": {
                "ticker": row["ticker"],
                "date": row["date"].isoformat(),
                "close": row["close"],
                "daily_return": row["daily_return"],
                "market_return": row["market_return"],
            },
        })
    return events


def delivery_report(err, msg):
    if err is not None:
        print(f"  delivery failed for {msg.key()}: {err}")


def main():
    filing_events = load_filing_events()
    earliest_filing = min(e["timestamp"] for e in filing_events)
    price_events = load_price_events(min_date=earliest_filing)

    events = filing_events + price_events
    events.sort(key=lambda e: e["timestamp"])
    print(f"Replaying {len(events)} events "
          f"({events[0]['timestamp'].date()} to {events[-1]['timestamp'].date()}) "
          f"at {config.REPLAY_SPEEDUP_FACTOR}x, capped at {REPLAY_MAX_STEP_DELAY_SECONDS}s/step.")

    producer = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS})

    prev_timestamp = events[0]["timestamp"]
    for i, event in enumerate(events):
        gap_seconds = (event["timestamp"] - prev_timestamp).total_seconds()
        delay = min(gap_seconds / config.REPLAY_SPEEDUP_FACTOR, REPLAY_MAX_STEP_DELAY_SECONDS)
        if delay > 0:
            time.sleep(delay)
        prev_timestamp = event["timestamp"]

        producer.produce(
            event["topic"],
            key=event["key"],
            value=json.dumps(event["value"], default=str),
            callback=delivery_report,
        )
        producer.poll(0)

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(events)} events replayed")

    producer.flush()
    print(f"\nDone. Replayed {len(events)} events onto "
          f"'{config.KAFKA_FILINGS_TOPIC}' and '{config.KAFKA_PRICES_TOPIC}'.")


if __name__ == "__main__":
    main()
