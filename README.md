# SEC Filing Market Reaction

Does the sentiment or materiality of an SEC 8-K filing predict how a
stock's price reacts? I built the full pipeline to find out: pull filings
and prices, score each filing with an LLM, run an event study, train a
model on it, then wrap the whole thing in a real-time streaming setup.

Short version of the answer: filings move prices a little on average, but
the sentiment score doesn't predict which way or how much. That's in the
results section below, reported straight rather than massaged into
something more exciting.

## Setup

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # venv\Scripts\pip on Windows
cp .env.example .env
```

Fill in `ANTHROPIC_API_KEY` in `.env` for the scoring step. If you want to
run the real-time pipeline against live data you'll also need
`ALPACA_API_KEY` / `ALPACA_API_SECRET` (free tier, real-time IEX feed).

Run everything from the repo root — `python 01_data_collection/fetch_filings.py`,
not `cd`'d into the folder.

## What's in each folder

`01_data_collection` pulls a year of 8-Ks for 40 tickers from EDGAR plus
daily prices and the SPY benchmark from yfinance, and joins them into one
dataset.

`02_nlp_scoring` runs each filing through Claude Haiku for a sentiment
score and a materiality rating, then checks the model against 40 filings
I labeled by hand before trusting the rest.

`03_event_study` is the actual test: a market-model regression per stock
against SPY, abnormal returns around each filing, significance tests on
the whole thing.

`04_predictive_model` trains a model on filing features to predict the
reaction, using time-based cross-validation (a random split would leak
future data into training, so that's not really optional for a time
series).

`05_realtime_pipeline` is Kafka + Docker Compose ingesting filings and
prices, a processor that predicts and then tracks the actual reaction as
it unfolds, and a Streamlit dashboard. There's a replay mode that runs
historical data through it on a sped-up clock, since live 8-Ks don't
exactly show up on a demo schedule.

## Results

528 filings, 40 tickers. The average abnormal return in the 3 days after
a filing is about -0.5% and it's statistically significant (p = 0.01) —
so something real is happening. But sentiment score and materiality
barely correlate with it on their own (r ≈ 0, p > 0.7), and even after
throwing more features at it in the predictive model — sector, filing
length, event type, each stock's historical beta — nothing beat just
guessing "no reaction" on data the model hadn't seen.

Take this as a first pass, not a final word — one year of data and one
LLM's sentiment scores isn't a lot to generalize from. But at least with
this setup, 8-K text doesn't seem to carry much signal about how the
stock moves afterward.

## Running everything

```bash
python 01_data_collection/fetch_filings.py
python 01_data_collection/fetch_prices.py
python 01_data_collection/build_dataset.py

python 02_nlp_scoring/score_filings.py
python 02_nlp_scoring/create_labeling_sample.py   # hand-label data/labeling_sample.csv, then:
python 02_nlp_scoring/validate_scoring.py

python 03_event_study/compute_abnormal_returns.py
python 03_event_study/run_significance_tests.py

python 04_predictive_model/build_features.py
python 04_predictive_model/train_and_evaluate.py
```

Real-time pipeline, each command in its own terminal:

```bash
docker compose up -d                                  # Kafka + kafka-ui, localhost:8080

python 05_realtime_pipeline/replay_producer.py         # or filing_producer.py + price_producer.py live
python 05_realtime_pipeline/reaction_processor.py
streamlit run 05_realtime_pipeline/dashboard.py        # localhost:8501
```

Replay costs nothing and needs no keys. Live mode needs Alpaca creds and
calls the Anthropic API per new filing found — a few cents for a short
run, same rate as the scoring step above.

## A couple of things worth knowing

SEC EDGAR gets a real User-Agent and a rate limit on every request —
don't strip that out if you touch the fetch scripts. Haiku does the bulk
scoring instead of a bigger model because it's cheap enough to run on
500+ filings, and the hand-labeling step exists specifically to check
that the cheaper model is actually good enough rather than just assuming
it. Everything here runs on free data (EDGAR, Alpaca's free tier) — no
paid APIs anywhere in the pipeline.
