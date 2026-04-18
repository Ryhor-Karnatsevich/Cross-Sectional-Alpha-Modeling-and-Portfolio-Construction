import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Data")

RAW_PRICES_PATH = os.path.join(DATA_DIR, "Raw", "prices.parquet")
RETURNS_PATH = os.path.join(DATA_DIR, "Processed", "returns.parquet")
PRICES_LONG_PATH = os.path.join(DATA_DIR, "Processed", "prices_long.parquet")
AVAILABILITY_PATH = os.path.join(DATA_DIR, "Processed", "availability.parquet")
UNIVERSE_PATH = os.path.join(DATA_DIR, "Raw", "universe.csv")
VOLUME_PATH = RAW_PRICES_PATH.replace("prices", "volume")
LIQUIDITY_PATH = RAW_PRICES_PATH.replace("prices", "liquidity")
FORWARD_RETURNS_PATH = os.path.join(DATA_DIR, "Processed", "forward_returns.parquet")

START_DATE = "2010-01-01"
MIN_COVERAGE = 0.8


