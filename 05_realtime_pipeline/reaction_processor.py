"""
Consumes the filings and prices topics, predicts each new filing's
reaction using the persisted model, then tracks realized abnormal returns
as subsequent price ticks arrive and writes both to a local SQLite table
the dashboard reads.

Beta and alpha come from each ticker's most recent historical estimate in
abnormal_returns.parquet, not a live regression refit. A handful of
intraday ticks isn't a meaningful substitute for the 250-daily-close
window that estimate was originally fit on, so this reuses the static
per-ticker value instead of pretending to re-estimate it live.

Run this AFTER compute_abnormal_returns.py and train_and_evaluate.py have
produced abnormal_returns.parquet and model_pipeline.joblib, and after
either replay_producer.py or the live producers are publishing to Kafka.

Usage:
    python reaction_processor.py
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from confluent_kafka import Consumer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "04_predictive_model"))
import config
from build_features import add_item_type_flags
from train_and_evaluate import ALL_FEATURES

WINDOW_LENGTH = config.REACTION_WINDOW_TRADING_DAYS + 1  # day 0 through day +N, inclusive


def load_beta_lookup() -> dict:
    returns = pd.read_parquet(config.ABNORMAL_RETURNS_PATH)
    returns = returns.sort_values("filing_date")
    latest = returns.groupby("ticker").last()
    return {ticker: (row["alpha"], row["beta"]) for ticker, row in latest.iterrows()}


def init_db():
    conn = sqlite3.connect(config.REALTIME_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reactions (
            ticker TEXT,
            accession_number TEXT,
            filing_date TEXT,
            predicted_car REAL,
            realized_car REAL,
            days_seen INTEGER,
            status TEXT,
            updated_at TEXT,
            PRIMARY KEY (ticker, accession_number)
        )
    """)
    conn.commit()
    return conn


def upsert_reaction(conn, ticker, accession_number, filing_date, predicted_car, realized_car, days_seen, status):
    conn.execute("""
        INSERT OR REPLACE INTO reactions
        (ticker, accession_number, filing_date, predicted_car, realized_car, days_seen, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, accession_number, filing_date, predicted_car, realized_car, days_seen, status,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()


def build_feature_row(message: dict, beta_lookup: dict) -> pd.DataFrame:
    ticker = message["ticker"]
    alpha, beta = beta_lookup.get(ticker, (0.0, 1.0))

    row = pd.DataFrame([{
        "sector": config.TICKER_SECTOR_MAP.get(ticker, "Misc large-cap"),
        "sentiment_score": message["sentiment_score"],
        "materiality": message["materiality"],
        "beta": beta,
        "char_count": message["char_count"],
        "item_type_summary": message["item_type_summary"],
    }])
    row = add_item_type_flags(row)
    return row[ALL_FEATURES]


def handle_filing(message: dict, model, beta_lookup: dict, pending: dict, conn):
    ticker = message["ticker"]
    features = build_feature_row(message, beta_lookup)
    predicted_car = float(model.predict(features)[0])

    pending.setdefault(ticker, []).append({
        "accession_number": message["accession_number"],
        "filing_date": message["filing_date"],
        "predicted_car": predicted_car,
        "realized_car": 0.0,
        "days_seen": 0,
    })

    upsert_reaction(conn, ticker, message["accession_number"], message["filing_date"],
                     predicted_car, None, 0, "pending")
    print(f"  [{ticker}] new filing {message['accession_number']}: predicted CAR = {predicted_car * 100:.3f}%")


def handle_price(message: dict, beta_lookup: dict, pending: dict, conn):
    ticker = message["ticker"]
    if ticker not in pending or not pending[ticker]:
        return

    alpha, beta = beta_lookup.get(ticker, (0.0, 1.0))
    abnormal_return = message["daily_return"] - (alpha + beta * message["market_return"])

    still_pending = []
    for entry in pending[ticker]:
        if message["date"] < entry["filing_date"]:
            still_pending.append(entry)
            continue

        entry["realized_car"] += abnormal_return
        entry["days_seen"] += 1
        status = "closed" if entry["days_seen"] >= WINDOW_LENGTH else "accumulating"

        upsert_reaction(conn, ticker, entry["accession_number"], entry["filing_date"],
                         entry["predicted_car"], entry["realized_car"], entry["days_seen"], status)

        if status == "closed":
            gap = entry["realized_car"] - entry["predicted_car"]
            print(f"  [{ticker}] window closed for {entry['accession_number']}: "
                  f"predicted = {entry['predicted_car'] * 100:.3f}%, "
                  f"realized = {entry['realized_car'] * 100:.3f}% (gap {gap * 100:.3f} pts)")
        else:
            still_pending.append(entry)

    pending[ticker] = still_pending


def main():
    model = joblib.load(config.MODEL_PIPELINE_PATH)
    beta_lookup = load_beta_lookup()
    conn = init_db()
    pending = {}

    consumer = Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": config.KAFKA_CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([config.KAFKA_FILINGS_TOPIC, config.KAFKA_PRICES_TOPIC])

    print(f"Loaded model from {config.MODEL_PIPELINE_PATH}, "
          f"beta lookup for {len(beta_lookup)} tickers.")
    print(f"Consuming '{config.KAFKA_FILINGS_TOPIC}' and '{config.KAFKA_PRICES_TOPIC}'. "
          f"Writing to {config.REALTIME_DB_PATH}. Ctrl+C to stop.")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"  consumer error: {msg.error()}")
                continue

            payload = json.loads(msg.value())
            if msg.topic() == config.KAFKA_FILINGS_TOPIC:
                handle_filing(payload, model, beta_lookup, pending, conn)
            elif msg.topic() == config.KAFKA_PRICES_TOPIC:
                handle_price(payload, beta_lookup, pending, conn)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    main()
