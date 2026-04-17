import yfinance as yf
import pandas as pd
import os
from config import RAW_PATH, RETURNS_PATH, START_DATE


# -------------------------
# DOWNLOAD
# -------------------------
def download_data(tickers, start=START_DATE, batch_size=50):
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        data = yf.download(
            tickers=batch,
            start=start,
            auto_adjust=True,
            progress=False
        )

        all_data.append(data)

    data = pd.concat(all_data, axis=1)

    # remove duplicated columns if any
    data = data.loc[:, ~data.columns.duplicated()]

    return data


# -------------------------
# ALIGNMENT
# -------------------------
def get_price_matrix(data):
    close = data["Close"].copy()
    close = close.sort_index()
    close = close.dropna(how="all")
    return close


# -------------------------
# CLEANING
# -------------------------
def clean_data(prices):
    prices = prices.sort_index()

    # limited forward fill
    prices = prices.ffill(limit=5)

    # keep rows with enough assets
    min_assets = int(0.5 * prices.shape[1])
    prices = prices.dropna(thresh=min_assets)

    return prices


# -------------------------
# RETURNS
# -------------------------
def compute_returns(prices):
    returns = prices.pct_change()
    return returns


# -------------------------
# STORAGE
# -------------------------
def save_raw(prices):
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    prices.to_parquet(RAW_PATH)


def save_returns(returns):
    os.makedirs(os.path.dirname(RETURNS_PATH), exist_ok=True)
    returns.to_parquet(RETURNS_PATH)


def load_raw():
    if os.path.exists(RAW_PATH):
        return pd.read_parquet(RAW_PATH)
    return None


def load_returns():
    if os.path.exists(RETURNS_PATH):
        return pd.read_parquet(RETURNS_PATH)
    return None


# -------------------------
# PIPELINE
# -------------------------
def build_and_save_dataset(tickers):
    raw = download_data(tickers)

    prices = get_price_matrix(raw)
    prices = clean_data(prices)

    returns = compute_returns(prices)

    save_raw(prices)
    save_returns(returns)

    return prices, returns