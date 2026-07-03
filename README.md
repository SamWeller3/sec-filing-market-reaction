# Phase 1 — Data Collection

Goal: build a clean, static dataset of 8-K filings + matching price history,
so Phase 2 (NLP scoring) and Phase 3 (event study + predictive model) have
something solid to work on. Nothing here is real-time yet — that's Phase 5,
after the science is validated.

## Setup

```bash
pip install -r requirements.txt
```

## Run, in order

```bash
python fetch_filings.py    # pulls every 8-K for your basket, downloads the text
python fetch_prices.py     # pulls daily price history + SPY benchmark
python build_dataset.py    # joins them, drops unusable rows, prints a summary
```

`fetch_filings.py` is the slow one — it's making one HTTP request per filing
to stay within SEC's rate-limit guidance, so expect it to take a while with
~40 tickers over a year. It's safe to re-run; it skips filings it's already
downloaded.

## What you'll have at the end

`data/phase1_master.parquet` — one row per usable 8-K filing event, with a
path to the filing's raw text and confirmation that there's enough price
history before/after it to run the Phase 3 event study.

## Before you touch Phase 2, sanity-check the output

- How many filings did you actually get? (printed at the end of `build_dataset.py`)
- Is any single ticker or time period wildly overrepresented?
- Open a couple of the raw filing text files in `data/raw_filings/` — do they
  look like real, readable 8-K content, or is the HTML-to-text conversion
  leaving in garbage (nav menus, XBRL tags, etc.)? If it's messy, that's worth
  fixing now — bad text in means bad sentiment scores out in Phase 2.

## Known things to adjust

- `config.SEC_USER_AGENT` is set to your name/email, which is what SEC
  actually wants (a real identifying contact, not a placeholder) — leave it.
- `TICKERS` is a starter basket of 40 diversified large-caps. Feel free to
  swap in others, just keep it diversified across sectors so you're not
  accidentally studying "how Big Tech reacts to news" only.
- `fetch_prices.py` uses yfinance for this historical pull since it needs no
  signup. When you get to Phase 5 (real-time), that's where Alpaca comes in.
