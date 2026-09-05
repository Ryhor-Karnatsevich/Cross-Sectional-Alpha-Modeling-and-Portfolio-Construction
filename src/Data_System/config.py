import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Data")
YFINANCE_CACHE_PATH = os.path.join(DATA_DIR, "Cache", "yfinance")

RAW_PRICES_PATH = os.path.join(DATA_DIR, "Raw", "prices.parquet")
RETURNS_PATH = os.path.join(DATA_DIR, "Processed", "returns.parquet")
PRICES_LONG_PATH = os.path.join(DATA_DIR, "Processed", "prices_long.parquet")
AVAILABILITY_PATH = os.path.join(DATA_DIR, "Processed", "availability.parquet")
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

START_DATE = "2010-01-01"
MIN_COVERAGE = 0.8
MAX_ABS_DAILY_RETURN = 1.0
MAX_EXTREME_DAILY_RETURNS = 1


