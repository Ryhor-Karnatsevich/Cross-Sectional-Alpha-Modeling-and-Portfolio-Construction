import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Data")
YFINANCE_CACHE_PATH = os.path.join(DATA_DIR, "Cache", "yfinance")

RAW_PRICES_PATH = os.path.join(DATA_DIR, "Raw", "prices.parquet")
RETURNS_PATH = os.path.join(DATA_DIR, "Processed", "returns.parquet")
PRICES_LONG_PATH = os.path.join(DATA_DIR, "Processed", "prices_long.parquet")
AVAILABILITY_PATH = os.path.join(DATA_DIR, "Processed", "availability.parquet")
QUALITY_PATH = os.path.join(DATA_DIR, "Processed", "data_quality.parquet")
MEMBERSHIP_PATH = os.path.join(DATA_DIR, "Processed", "membership.parquet")
UNIVERSE_PATH = os.path.join(DATA_DIR, "Raw", "universe.csv")
HISTORICAL_COMPONENTS_PATH = os.path.join(DATA_DIR, "Raw", "sp500_historical_components.csv")
VOLUME_PATH = RAW_PRICES_PATH.replace("prices", "volume")
LIQUIDITY_PATH = RAW_PRICES_PATH.replace("prices", "liquidity")
FORWARD_RETURNS_PATH = os.path.join(DATA_DIR, "Processed", "forward_returns.parquet")
RISK_FREE_RATE_PATH = os.path.join(DATA_DIR, "Raw", "dgs3mo.parquet")

HISTORICAL_COMPONENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)
FRED_DGS3MO_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

DATA_START_DATE = "2008-01-01"
SUSPICIOUS_ABS_DAILY_RETURN = 0.5
MAX_ABS_DAILY_RETURN = 1.0
ROUND_TRIP_RETURN_TOLERANCE = 0.25

# Manually verified market moves are kept even when they cross the automatic
# extreme-return threshold. Each entry is (ticker, date).
CONFIRMED_REAL_RETURN_EVENTS = {
    ("HIG", "2008-12-05"),
}

# Yahoo currently returns another security or a clearly corrupted series under
# these obsolete historical symbols. Until a verified continuous alias or a
# second data source is available, missing data is safer than false history.
YAHOO_REUSED_TICKERS = {
    "BMC",
    "CBE",
    "CFC",
    "COL",
    "CPWR",
    "EP",
    "EQ",
    "GR",
    "HPC",
    "MEE",
    "MI",
    "NCC",
    "PARA",
    "PBG",
    "PTV",
    "STI",
    "TIE",
}

# Only direct company/ticker renames are allowed here. Acquisitions, mergers
# without a clearly continuous security and post-bankruptcy tickers are excluded.
YAHOO_TICKER_ALIASES = {
    "ABC": "COR",
    "ADS": "BFH",
    "ANTM": "ELV",
    "BLL": "BALL",
    "CDAY": "DAY",
    "CTL": "LUMN",
    "FB": "META",
    "FBHS": "FBIN",
    "FII": "FHI",
    "FLT": "CPAY",
    "KORS": "CPRI",
    "NLOK": "GEN",
    "PKI": "RVTY",
    "SYMC": "GEN",
    "TMK": "GL",
    "WLTW": "WTW",
}


