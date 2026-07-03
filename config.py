"""
Phase 1 configuration: the company basket, date range, and SEC access settings.
"""

# --- SEC identifies you by this header on every request. This is not optional --
# SEC will rate-limit/block you if you use a generic or missing User-Agent.
# Format they ask for: "Sample Company Name AdminContact@sample.com"
SEC_USER_AGENT = "Sam Weller samoweller@comcast.net"

# SEC asks for no more than ~10 requests/second. We stay well under that.
SEC_REQUEST_DELAY_SECONDS = 0.2

# How far back to pull 8-K filings.
LOOKBACK_DAYS = 365

# A diversified basket across sectors -- avoids the whole dataset being one
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
MASTER_DATASET_PATH = f"{DATA_DIR}/phase1_master.parquet"
