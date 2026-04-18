import yfinance as yf
import pandas as pd
import numpy as np
import os

from config import (
    RAW_PRICES_PATH,
    RETURNS_PATH,
    PRICES_LONG_PATH,
    AVAILABILITY_PATH,
    UNIVERSE_PATH,
    START_DATE,
    VOLUME_PATH,
    LIQUIDITY_PATH,
    FORWARD_RETURNS_PATH,
    MIN_COVERAGE
)

# IMPORTANT:
# All future features must be computed using data up to t-1
# returns represent t → t+1


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


def get_volume_matrix(data):
    volume = data["Volume"].copy()
    volume = volume.sort_index()
    volume = volume.dropna(how="all")
    return volume


# -------------------------
# CLEANING
# -------------------------
def clean_data(prices):
    prices = prices.sort_index()

    # removed unstable outlier logic → rely on return clipping instead
    prices = prices.ffill(limit=5)

    min_assets = int(0.5 * prices.shape[1])
    prices = prices.dropna(thresh=min_assets)

    return prices


# -------------------------
# RETURNS
# -------------------------
def compute_returns(prices):
    returns = prices.pct_change()

    # robust clipping instead of price-level outlier removal
    returns = returns.clip(-0.5, 0.5)
    clipped = ((returns == 0.5) | (returns == -0.5)).sum().sum()
    print(f"Clipped returns count: {clipped}")

    returns = returns.where(prices.notna())
    return returns


# -------------------------
# LIQUIDITY PROXY
# -------------------------
def compute_liquidity(prices, volume):
    dollar_volume = prices * volume

    # FIX: stabilize heavy-tailed distribution
    liquidity = np.log1p(dollar_volume.rolling(20).mean())

    return liquidity


# -------------------------
# AUX DATA
# -------------------------
def compute_availability(prices):
    return ~prices.isna()


def to_long(prices):
    return (
        prices
        .stack()
        .reset_index()
        .rename(columns={"level_1": "ticker", 0: "price"})
    )


# -------------------------
# STORAGE
# -------------------------
def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_forward_returns(prices):
    fwd = prices.pct_change().shift(-1)

    # FIX: correct masking → require price at t+1
    fwd = fwd.where(prices.shift(-1).notna())

    return fwd


def save_all(prices, returns, volume, liquidity, prices_long, availability, forward_returns, tickers):
    for path in [
        RAW_PRICES_PATH,
        RETURNS_PATH,
        PRICES_LONG_PATH,
        AVAILABILITY_PATH,
        VOLUME_PATH,
        LIQUIDITY_PATH,
        UNIVERSE_PATH,
        FORWARD_RETURNS_PATH
    ]:
        _ensure_dir(path)

    prices.to_parquet(RAW_PRICES_PATH)
    returns.to_parquet(RETURNS_PATH)
    forward_returns.to_parquet(FORWARD_RETURNS_PATH)
    volume.to_parquet(VOLUME_PATH)
    liquidity.to_parquet(LIQUIDITY_PATH)
    prices_long.to_parquet(PRICES_LONG_PATH)
    availability.to_parquet(AVAILABILITY_PATH)

    pd.Series(tickers).to_csv(UNIVERSE_PATH, index=False)




# -------------------------
# CHECKS
# -------------------------
def sanity_checks(prices, volume):
    assert prices.index.is_monotonic_increasing
    assert prices.shape[1] > 100
    assert prices.index.equals(volume.index)
    assert prices.columns.equals(volume.columns)

    if (volume < 0).any().any():
        raise ValueError("Negative volume detected")

    # duplicate dates check
    if prices.index.duplicated().any():
        dupes = prices.index[prices.index.duplicated()]
        raise ValueError(f"Duplicate dates found: {dupes[:5]}")

    print("Volume NaN ratio:", volume.isna().mean().mean())


# ----------------------------------------------
# for data uncertainty in volume
def clean_volume(volume):
    volume = volume.astype(float)
    volume = volume.mask(volume < 0)
    return volume


# -------------------------
# UNIVERSE FILTER
# -------------------------
def filter_universe(prices, liquidity, min_assets=150):
    valid_counts = prices.notna().sum(axis=1)
    mask = valid_counts >= min_assets

    prices = prices.loc[mask]
    liquidity = liquidity.loc[mask]

    return prices, liquidity

# --------------------------------
# gaps check
def check_extreme_gaps(prices, max_gap=5):
    max_gaps = {}

    for col in prices.columns:
        is_nan = prices[col].isna().astype(int)

        groups = (is_nan != is_nan.shift()).cumsum()
        gap_lengths = is_nan.groupby(groups).cumsum()

        max_gaps[col] = gap_lengths.max()

    max_gaps = pd.Series(max_gaps)

    problematic = max_gaps[max_gaps > max_gap]

    if len(problematic) > 0:
        print(f"Warning: {len(problematic)} tickers have gaps > {max_gap}")
        print(problematic.sort_values(ascending=False).head())
        print(problematic.sort_values(ascending=False).tail())


# -------------------------
# BUILD
# -------------------------
def build_and_save_dataset(tickers):
    raw = download_data(tickers)

    prices = get_price_matrix(raw)
    volume = get_volume_matrix(raw)

    prices = clean_data(prices)

    volume = volume.loc[prices.index, prices.columns]
    volume = volume.where(prices.notna())

    volume = clean_volume(volume)
    volume = volume.astype(float)

    returns = compute_returns(prices)
    forward_returns = compute_forward_returns(prices)
    liquidity = compute_liquidity(prices, volume)

    # -------------------------
    # COVERAGE FILTER (ASSET LEVEL)
    # -------------------------
    coverage = prices.notna().mean()
    valid_assets = coverage >= MIN_COVERAGE

    prices = prices.loc[:, valid_assets]
    volume = volume.loc[:, valid_assets]
    returns = returns.loc[:, valid_assets]
    forward_returns = forward_returns.loc[:, valid_assets]
    liquidity = liquidity.loc[:, valid_assets]

    deleted = (~valid_assets).sum()
    print(f"deleted tickers: {deleted}")

    # FIX: explicit effective universe
    print(f"Effective universe size: {prices.shape[1]}")

    # -------------------------
    # UNIVERSE FILTER (TIME LEVEL)
    # -------------------------
    prices, liquidity = filter_universe(prices, liquidity)

    returns = returns.loc[prices.index]
    forward_returns = forward_returns.loc[prices.index]
    volume = volume.loc[prices.index]

    availability = compute_availability(prices)
    prices_long = to_long(prices)

    sanity_checks(prices, volume)
    check_extreme_gaps(prices)
    print(returns.std().describe())
    print(forward_returns.std().describe())

    save_all(
        prices,
        returns,
        volume,
        liquidity,
        prices_long,
        availability,
        forward_returns,
        tickers
    )

    return prices, returns, volume, liquidity, prices_long, availability, forward_returns


# -------------------------
# PIPELINE
# -------------------------
def run_pipeline():
    paths = [
        RAW_PRICES_PATH,
        RETURNS_PATH,
        PRICES_LONG_PATH,
        AVAILABILITY_PATH,
        VOLUME_PATH,
        LIQUIDITY_PATH,
        FORWARD_RETURNS_PATH
    ]

    dataset_exists = all(os.path.exists(p) for p in paths)

    if dataset_exists:
        print("Dataset found -> loading")

        prices = pd.read_parquet(RAW_PRICES_PATH)
        returns = pd.read_parquet(RETURNS_PATH)
        volume = pd.read_parquet(VOLUME_PATH)
        liquidity = pd.read_parquet(LIQUIDITY_PATH)
        prices_long = pd.read_parquet(PRICES_LONG_PATH)
        availability = pd.read_parquet(AVAILABILITY_PATH)
        forward_returns = pd.read_parquet(FORWARD_RETURNS_PATH)

    else:
        print("Dataset missing -> rebuilding")

        from get_tickers import get_sp500_tickers
        tickers = get_sp500_tickers()

        prices, returns, volume, liquidity, prices_long, availability, forward_returns = build_and_save_dataset(tickers)

    return prices, returns, volume, liquidity, prices_long, availability, forward_returns



# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    prices, returns, volume, liquidity, prices_long, availability, forward_returns = run_pipeline()

    print("\nShapes:")
    print("Prices:", prices.shape)
    print("Returns:", returns.shape)
    print("Volume:", volume.shape)
    print("Liquidity:", liquidity.shape)
    print("Long:", prices_long.shape)
    print("Forward Returns:", forward_returns.shape)

