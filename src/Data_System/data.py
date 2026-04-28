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


# -------------------------------------------------------------------------------------------------
# DOWNLOAD
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
# -------------------------------------------------------------------------------------------------


# DATA CALCULATING
# -------------------------------------------------------------------------------------------------
# PRICES
def get_price_matrix(data):
    prices = data["Close"].copy()
    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    # Fill forward missing values (max 5 periods)
    prices = prices.ffill(limit=5)

    # Drop dates where less than 50% of assets have data
    min_assets = int(0.5 * prices.shape[1])
    prices = prices.dropna(thresh=min_assets)

    return prices


# VOLUME
def get_volume_matrix(data,prices):
    volume = data["Volume"].copy()
    volume = volume.sort_index()
    volume = volume.dropna(how="all")

    volume = volume.astype(float)
    # Hide invalid data
    volume = volume.mask(volume < 0)
    return volume


# RETURNS
def compute_returns(prices):
    returns = prices.pct_change()

    # robust clipping instead of price-level outlier removal
    returns = returns.clip(-0.5, 0.5)
    clipped = ((returns == 0.5) | (returns == -0.5)).sum().sum()
    print(f"Clipped returns count: {clipped}")

    # create returns only for existing prices
    returns = returns.where(prices.notna())
    return returns


# LIQUIDITY
def compute_liquidity(prices, volume):
    dollar_volume = prices * volume
    # To stabilize heavy-tailed distribution. log(1+x) to avoid errors with 0 values.
    liquidity = np.log1p(dollar_volume.rolling(20).mean())
    return liquidity


# Long prices
def to_long(prices):
    return (
        prices
        .stack()
        .reset_index()
        .rename(columns={"level_1": "ticker", 0: "price"})
    )


# Forward Returns
def compute_forward_returns(prices, horizon=21):
    fwd = prices.pct_change(horizon).shift(-horizon)
    return fwd


# Needed to created availability dataset for prices
def compute_availability(prices):
    return ~prices.isna()
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# CHECKS
def sanity_checks(prices, volume):
    assert prices.index.is_monotonic_increasing     # check if indexes going straight
    assert prices.shape[1] > 100                    # check if there are more than 100 columns
    assert prices.index.equals(volume.index)            # matches two datasets
    assert prices.columns.equals(volume.columns)

    if (volume < 0).any().any():                        # negative volume test
        raise ValueError("Negative volume detected")

    # duplicate dates check
    if prices.index.duplicated().any():                 # duplicates test
        dupes = prices.index[prices.index.duplicated()]
        raise ValueError(f"Duplicate dates found: {dupes[:5]}")

    print("Volume NaN ratio:", volume.isna().mean().mean())  # count ratio of missing values
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# UNIVERSE FILTER
def filter_universe(prices, liquidity, min_assets=150):
    initial_days = len(prices)

    valid_counts = prices.notna().sum(axis=1)
    mask = valid_counts >= min_assets

    prices_filtered = prices.loc[mask]
    liquidity_filtered = liquidity.loc[mask]

    dropped_days = initial_days - len(prices_filtered)
    if dropped_days > 0:
        print(f"--- Universe Filter Applied ---")
        print(f"Dropped {dropped_days} days due to low asset count (min_assets={min_assets})")
        print(f"Remaining days: {len(prices_filtered)}")
    return prices_filtered, liquidity_filtered
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# Gaps check
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
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# STORAGE
def save_all(prices, returns, volume, liquidity, prices_long, availability, forward_returns, tickers):
    paths = {
        RAW_PRICES_PATH: prices,
        RETURNS_PATH: returns,
        FORWARD_RETURNS_PATH: forward_returns,
        VOLUME_PATH: volume,
        LIQUIDITY_PATH: liquidity,
        PRICES_LONG_PATH: prices_long,
        AVAILABILITY_PATH: availability,
    }

    for path, df in paths.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)

    # universe
    os.makedirs(os.path.dirname(UNIVERSE_PATH), exist_ok=True)
    pd.Series(tickers).to_csv(UNIVERSE_PATH, index=False)
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# BUILD
def build_and_save_dataset(tickers):
    raw = download_data(tickers)

    prices = get_price_matrix(raw)
    volume = get_volume_matrix(raw)

    # Align volume based on prices
    volume = volume.loc[prices.index, prices.columns]
    volume = volume.where(prices.notna())

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
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# PIPELINE
def run_pipeline():
    paths = [
        RAW_PRICES_PATH,
        RETURNS_PATH,
        VOLUME_PATH,
        LIQUIDITY_PATH,
        PRICES_LONG_PATH,
        AVAILABILITY_PATH,
        FORWARD_RETURNS_PATH
    ]

    dataset_exists = all(os.path.exists(p) for p in paths)

    if dataset_exists:
        print("Dataset found -> loading")
        return tuple(pd.read_parquet(p) for p in paths)

    print("Dataset missing -> rebuilding")

    from get_tickers import get_sp500_tickers
    return build_and_save_dataset(get_sp500_tickers())
# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# ENTRY
if __name__ == "__main__":
    prices, returns, volume, liquidity, prices_long, availability, forward_returns = run_pipeline()

    print("\nShapes:")
    print("Prices:", prices.shape)
    print("Returns:", returns.shape)
    print("Volume:", volume.shape)
    print("Liquidity:", liquidity.shape)
    print("Long:", prices_long.shape)
    print("Forward Returns:", forward_returns.shape)

