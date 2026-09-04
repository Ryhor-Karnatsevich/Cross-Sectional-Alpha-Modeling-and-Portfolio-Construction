import yfinance as yf
import pandas as pd
import numpy as np
import os

from config import (
    RAW_PRICES_PATH,
    RETURNS_PATH,
    PRICES_LONG_PATH,
    AVAILABILITY_PATH,
    MEMBERSHIP_PATH,
    UNIVERSE_PATH,
    HISTORICAL_COMPONENTS_PATH,
    YFINANCE_CACHE_PATH,
    START_DATE,
    VOLUME_PATH,
    LIQUIDITY_PATH,
    FORWARD_RETURNS_PATH,
    MIN_COVERAGE,
    MAX_ABS_DAILY_RETURN,
    MAX_EXTREME_DAILY_RETURNS,
)

# IMPORTANT:
# All future features must be computed using data up to t-1
# returns represent t → t+1


# -------------------------------------------------------------------------------------------------
# DOWNLOAD
def download_data(tickers, start=START_DATE, batch_size=50):
    os.makedirs(YFINANCE_CACHE_PATH, exist_ok=True)
    yf.set_tz_cache_location(YFINANCE_CACHE_PATH)

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

    if data.empty or data["Close"].dropna(how="all").empty:
        raise RuntimeError("yfinance returned no price data")

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

    return prices


# VOLUME
def get_volume_matrix(data):
    volume = data["Volume"].copy()
    volume = volume.sort_index()
    volume = volume.dropna(how="all")

    volume = volume.astype(float)
    # Hide invalid data
    volume = volume.mask(volume < 0)
    return volume


# RETURNS
def compute_returns(prices, membership=None):
    returns = prices.pct_change()

    # robust clipping instead of price-level outlier removal
    clipped_mask = returns.abs() >= 0.5
    returns = returns.clip(-0.5, 0.5)
    if membership is not None:
        clipped_mask &= membership
    clipped = clipped_mask.sum().sum()
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
def compute_availability(prices, membership):
    return prices.notna() & membership
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
def filter_universe(prices, liquidity, membership, min_assets=150):
    initial_days = len(prices)

    valid_counts = (prices.notna() & membership).sum(axis=1)
    mask = valid_counts >= min_assets

    prices_filtered = prices.loc[mask]
    liquidity_filtered = liquidity.loc[mask]
    membership_filtered = membership.loc[mask]

    dropped_days = initial_days - len(prices_filtered)
    if dropped_days > 0:
        print(f"--- Universe Filter Applied ---")
        print(f"Dropped {dropped_days} days due to low asset count (min_assets={min_assets})")
        print(f"Remaining days: {len(prices_filtered)}")
    return prices_filtered, liquidity_filtered, membership_filtered
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# Gaps check
def check_extreme_gaps(prices, membership, max_gap=5):
    max_gaps = {}

    for col in prices.columns:
        is_nan = (prices[col].isna() & membership[col]).astype(int)

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
def save_all(
    prices,
    returns,
    volume,
    liquidity,
    prices_long,
    availability,
    membership,
    forward_returns,
    universe_report,
):
    paths = {
        RAW_PRICES_PATH: prices,
        RETURNS_PATH: returns,
        FORWARD_RETURNS_PATH: forward_returns,
        VOLUME_PATH: volume,
        LIQUIDITY_PATH: liquidity,
        PRICES_LONG_PATH: prices_long,
        AVAILABILITY_PATH: availability,
        MEMBERSHIP_PATH: membership,
    }

    for path, df in paths.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)

    # universe
    os.makedirs(os.path.dirname(UNIVERSE_PATH), exist_ok=True)
    universe_report.to_csv(UNIVERSE_PATH, index=False)
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# BUILD
def build_and_save_dataset(history, tickers):
    raw = download_data(tickers)

    prices = get_price_matrix(raw).reindex(columns=tickers)
    volume = get_volume_matrix(raw).reindex(columns=tickers)

    latest_membership_date = history["date"].max()
    prices = prices.loc[:latest_membership_date]
    volume = volume.loc[:latest_membership_date]

    from get_tickers import build_membership_matrix

    membership = build_membership_matrix(history, prices.index, prices.columns)

    # Align volume based on prices
    volume = volume.reindex(index=prices.index, columns=prices.columns)
    volume = volume.where(prices.notna())

    # -------------------------
    # COVERAGE FILTER (ASSET LEVEL)
    # -------------------------
    member_observations = membership.sum()
    available_member_observations = (prices.notna() & membership).sum()
    coverage = available_member_observations.div(member_observations).fillna(0)
    raw_daily_returns = prices.pct_change(fill_method=None)
    extreme_daily_returns = (
        (raw_daily_returns.abs() > MAX_ABS_DAILY_RETURN) & membership
    ).sum()
    repeated_extreme_returns = extreme_daily_returns > MAX_EXTREME_DAILY_RETURNS
    valid_assets = (coverage >= MIN_COVERAGE) & ~repeated_extreme_returns

    exclusion_reason = pd.Series("included", index=prices.columns)
    exclusion_reason.loc[coverage < MIN_COVERAGE] = "price_coverage_below_threshold"
    exclusion_reason.loc[repeated_extreme_returns] = "repeated_extreme_daily_returns"

    first_membership = membership.apply(
        lambda column: column.index[column.argmax()] if column.any() else pd.NaT
    )
    last_membership = membership.apply(
        lambda column: column.index[len(column) - 1 - column.iloc[::-1].argmax()]
        if column.any()
        else pd.NaT
    )

    universe_report = pd.DataFrame({
        "ticker": prices.columns,
        "first_membership_date": first_membership.reindex(prices.columns).values,
        "last_membership_date": last_membership.reindex(prices.columns).values,
        "membership_observations": member_observations.reindex(prices.columns).values,
        "price_coverage_during_membership": coverage.reindex(prices.columns).values,
        "extreme_daily_returns": extreme_daily_returns.reindex(prices.columns).values,
        "included": valid_assets.reindex(prices.columns).values,
        "exclusion_reason": exclusion_reason.reindex(prices.columns).values,
    })

    prices = prices.loc[:, valid_assets]
    volume = volume.loc[:, valid_assets]
    membership = membership.loc[:, valid_assets]

    deleted = (~valid_assets).sum()
    print(f"deleted tickers: {deleted}")

    # FIX: explicit effective universe
    print(f"Effective universe size: {prices.shape[1]}")

    returns = compute_returns(prices, membership)
    forward_returns = compute_forward_returns(prices)
    liquidity = compute_liquidity(prices, volume)

    # -------------------------
    # UNIVERSE FILTER (TIME LEVEL)
    # -------------------------
    prices, liquidity, membership = filter_universe(prices, liquidity, membership)

    returns = returns.loc[prices.index]
    forward_returns = forward_returns.loc[prices.index]
    volume = volume.loc[prices.index]

    availability = compute_availability(prices, membership)
    prices_long = to_long(prices)

    sanity_checks(prices, volume)
    check_extreme_gaps(prices, membership)
    print(returns.std().describe())
    print(forward_returns.std().describe())

    save_all(
        prices,
        returns,
        volume,
        liquidity,
        prices_long,
        availability,
        membership,
        forward_returns,
        universe_report,
    )

    return prices, returns, volume, liquidity, prices_long, availability, forward_returns
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# PIPELINE
def run_pipeline():
    data_paths = [
        RAW_PRICES_PATH,
        RETURNS_PATH,
        VOLUME_PATH,
        LIQUIDITY_PATH,
        PRICES_LONG_PATH,
        AVAILABILITY_PATH,
        FORWARD_RETURNS_PATH,
    ]
    required_paths = data_paths + [
        MEMBERSHIP_PATH,
        UNIVERSE_PATH,
        HISTORICAL_COMPONENTS_PATH,
    ]

    dataset_exists = all(os.path.exists(path) for path in required_paths)

    if dataset_exists:
        print("Dataset found -> loading")
        return tuple(pd.read_parquet(path) for path in data_paths)

    print("Dataset missing -> rebuilding")

    from get_tickers import get_sp500_history, get_sp500_tickers

    history = get_sp500_history()
    tickers = get_sp500_tickers(history)

    print(f"Historical source snapshots: {len(history)}")
    print(f"Historical ticker union since {START_DATE}: {len(tickers)}")

    return build_and_save_dataset(history, tickers)
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

