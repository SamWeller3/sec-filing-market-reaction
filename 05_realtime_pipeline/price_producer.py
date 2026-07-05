"""
Subscribes to Alpaca's real-time IEX feed for the ticker basket plus SPY
as the market benchmark, and publishes minute-bar returns to the prices
topic. This is the live counterpart to replay_producer.py.

Methodological caveat: the rest of this project (the event study, the
predictive model) is built on daily closes and daily returns. Alpaca's
daily bars only close once at end of day, which would leave a short demo
session's live feed empty most of the time, so this uses minute bars and
a minute-over-minute return as an intraday stand-in instead. That's a
real approximation, not equivalent to the daily-return methodology used
everywhere else in this project -- good enough to make the pipeline
observable live, not a rigorous intraday event study. The message field
is still named daily_return so reaction_processor.py doesn't need
separate live/replay logic, even though in live mode it holds a minute
return, not a daily one.

Needs ALPACA_API_KEY and ALPACA_API_SECRET in .env.

Usage:
    python price_producer.py
"""

import json
import os
import sys

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from confluent_kafka import Producer
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

load_dotenv()

producer = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS})
previous_close = {}
latest_market_return = {"value": 0.0}


async def handle_bar(bar):
    symbol = bar.symbol
    prev = previous_close.get(symbol)
    previous_close[symbol] = bar.close
    if prev is None:
        return  # first bar for this symbol, nothing to compute a return against yet

    minute_return = (bar.close - prev) / prev

    if symbol == "SPY":
        latest_market_return["value"] = minute_return
        return

    message = {
        "ticker": symbol,
        "date": bar.timestamp.isoformat(),
        "close": bar.close,
        "daily_return": minute_return,
        "market_return": latest_market_return["value"],
    }
    producer.produce(config.KAFKA_PRICES_TOPIC, key=symbol, value=json.dumps(message))
    producer.poll(0)
    print(f"  [{symbol}] minute return {minute_return * 100:.3f}%")


def main():
    stream = StockDataStream(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_API_SECRET"],
        feed=DataFeed.IEX,
    )
    stream.subscribe_bars(handle_bar, *config.TICKERS, "SPY")
    print(f"Subscribed to live IEX minute bars for {len(config.TICKERS)} tickers plus SPY. Ctrl+C to stop.")
    stream.run()


if __name__ == "__main__":
    main()
