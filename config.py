"""
Project configuration: the company basket, date range, and SEC access settings.
"""

# SEC identifies you by this header on every request; it's not optional.
# SEC will rate-limit or block you if you use a generic or missing User-Agent.
# Format they ask for: "Sample Company Name AdminContact@sample.com"
SEC_USER_AGENT = "Sam Weller samoweller@comcast.net"

# SEC asks for no more than ~10 requests/second. We stay well under that.
SEC_REQUEST_DELAY_SECONDS = 0.2

# How far back to pull 8-K filings.
LOOKBACK_DAYS = 365

# A diversified basket across sectors, so the dataset isn't dominated by one
# industry's idiosyncratic news cycle. 40 large, liquid, well-covered names
# means clean price data and a steady stream of 8-Ks to work with.
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM", "ADBE",
    # Financials
    "JPM", "BAC", "GS", "MS", "SCHW", "BLK", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV",
    # Consumer
    "AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "BA", "CAT", "GE", "HON",
    # Comms / media
    "DIS", "NFLX", "T", "VZ",
    # Misc large-cap
    "TSLA", "V", "MA",
]

# Local storage layout
DATA_DIR = "data"
RAW_FILINGS_DIR = f"{DATA_DIR}/raw_filings"
FILINGS_METADATA_PATH = f"{DATA_DIR}/filings_metadata.parquet"
PRICES_PATH = f"{DATA_DIR}/prices.parquet"
MASTER_DATASET_PATH = f"{DATA_DIR}/master_dataset.parquet"

# NLP scoring

# Haiku, not Sonnet: this is bulk classification across 500+ filings, and
# the point of the labeling-sample validation step is to confirm a cheaper
# model is actually good enough here, rather than assuming it.
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# 8-K text (cover page + press release exhibit) gets truncated to this many
# characters before scoring, mostly to control cost. Sentiment/materiality
# doesn't need every line of a financial table, just the narrative content.
MAX_FILING_CHARS = 6000

FILING_SCORES_PATH = f"{DATA_DIR}/filing_scores.parquet"
LABELING_SAMPLE_PATH = f"{DATA_DIR}/labeling_sample.csv"
LABELING_SAMPLE_SIZE = 40
LABELING_RANDOM_SEED = 42

# Event study

# Market-model regression window: the 250 trading days strictly before
# filing_date. No gap before the event window, since 8-Ks aren't pre-scheduled
# like earnings, so there's no pre-announcement drift to buffer against.
ESTIMATION_WINDOW_TRADING_DAYS = 250

# CAR windows. Both stay inside the 5-trading-day floor every event has
# after its filing date, so nothing gets dropped or truncated.
EVENT_WINDOW_PRIMARY_DAYS = 2      # CAR over [0, +2]
EVENT_WINDOW_ROBUSTNESS_DAYS = 4   # CAR over [0, +4], robustness check

ABNORMAL_RETURNS_PATH = f"{DATA_DIR}/abnormal_returns.parquet"
EVENT_STUDY_RESULTS_PATH = f"{DATA_DIR}/event_study_results.csv"
EVENT_STUDY_CHART_PATH = f"{DATA_DIR}/event_study_chart.png"

# Predictive model

# Sector groupings reused from the TICKERS list above. Comments aren't
# accessible at runtime, so this needs to exist as real data if sector is
# going to be a feature.
TICKER_SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "NVDA": "Tech",
    "ORCL": "Tech", "CRM": "Tech", "ADBE": "Tech",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "SCHW": "Financials", "BLK": "Financials", "AXP": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare",
    "AMZN": "Consumer", "WMT": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "TGT": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "DIS": "Comms/media", "NFLX": "Comms/media", "T": "Comms/media", "VZ": "Comms/media",
    "TSLA": "Misc large-cap", "V": "Misc large-cap", "MA": "Misc large-cap",
}

MODEL_FEATURES_PATH = f"{DATA_DIR}/model_features.parquet"
MODEL_COMPARISON_CHART_PATH = f"{DATA_DIR}/model_comparison_chart.png"

CV_N_SPLITS = 5
HOLDOUT_FRACTION = 0.2   # chronological, most-recent 20% of events
RANDOM_SEED = 42         # model reproducibility only (e.g. RandomForest
                         # bootstrap sampling), never the train/test split
