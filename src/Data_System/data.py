import yfinance as yf
import pandas as pd
import numpy as np
import os

from config import (
    RAW_PRICES_PATH,
    RETURNS_PATH,
    PRICES_LONG_PATH,
    AVAILABILITY_PATH,
    QUALITY_PATH,
    MEMBERSHIP_PATH,
    UNIVERSE_PATH,
    YFINANCE_CACHE_PATH,
    DATA_START_DATE,
    VOLUME_PATH,
    VOLUME_QUALITY_PATH,
    LIQUIDITY_PATH,
    FORWARD_RETURNS_PATH,
    SUSPICIOUS_ABS_DAILY_RETURN,
    MAX_ABS_DAILY_RETURN,
    ROUND_TRIP_RETURN_TOLERANCE,
    CONFIRMED_REAL_RETURN_EVENTS,
    YAHOO_REUSED_TICKERS,
    YAHOO_TICKER_ALIASES,
)

from data_quality import (
    build_data_quality_mask,
    build_volume_quality_mask,
    first_true_date,
)

# IMPORTANT:
# All future features must be computed using data up to t-1
# returns represent t → t+1


# -------------------------------------------------------------------------------------------------
# DOWNLOAD
def download_yahoo_request(tickers, start, threads=True):
    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            auto_adjust=False,
            progress=False,
            threads=threads,
            multi_level_index=True,
        )
    except Exception as error:
        print(f"Yahoo request failed for {','.join(tickers)}: {error}")
        return pd.DataFrame()

    if data is None:
        return pd.DataFrame()
    return data


def ticker_has_prices(data, ticker):
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        return False

    price_column = ("Adj Close", ticker)
    return price_column in data.columns and data[price_column].notna().any()


def select_ticker_data(data, source_ticker, target_ticker):
    if not ticker_has_prices(data, source_ticker):
        return pd.DataFrame()

    columns = [column for column in data.columns if column[1] == source_ticker]
    selected = data.loc[:, columns].copy()
    selected.columns = pd.MultiIndex.from_tuples(
        [(column[0], target_ticker) for column in columns],
        names=data.columns.names,
    )
    return selected


def merge_downloads(data, additional_data):
    if data.empty:
        return additional_data.copy()
    if additional_data.empty:
        return data
    return data.combine_first(additional_data)


def replace_ticker_data(data, replacement, ticker):
    if data.empty:
        return replacement.copy()

    existing_columns = [column for column in data.columns if column[1] == ticker]
    if existing_columns:
        data = data.drop(columns=existing_columns)
    return merge_downloads(data, replacement)


def download_data(
    tickers,
    start=DATA_START_DATE,
    batch_size=50,
    ticker_aliases=None,
    reused_tickers=None,
):
    os.makedirs(YFINANCE_CACHE_PATH, exist_ok=True)
    yf.set_tz_cache_location(YFINANCE_CACHE_PATH)

    tickers = list(dict.fromkeys(tickers))
    if ticker_aliases is None:
        ticker_aliases = YAHOO_TICKER_ALIASES
    if reused_tickers is None:
        reused_tickers = YAHOO_REUSED_TICKERS

    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_data = download_yahoo_request(batch, start, threads=True)
        if not batch_data.empty:
            all_data.append(batch_data)

    if all_data:
        data = pd.concat(all_data, axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
    else:
        data = pd.DataFrame()

    initial_missing = [
        ticker for ticker in tickers if not ticker_has_prices(data, ticker)
    ]
    methods = {
        ticker: {
            "yahoo_ticker": ticker,
            "download_method": "batch" if ticker not in initial_missing else "missing",
        }
        for ticker in tickers
    }

    print(f"Initial batch tickers without prices: {len(initial_missing)}")

    for ticker in initial_missing:
        retry_data = download_yahoo_request([ticker], start, threads=False)
        retry_data = select_ticker_data(retry_data, ticker, ticker)

        if not retry_data.empty:
            data = merge_downloads(data, retry_data)
            methods[ticker]["download_method"] = "individual_retry"

    # Explicit aliases override the old symbol even when Yahoo has reused it
    # for a different security after the historical constituent changed ticker.
    for ticker in tickers:
        alias = ticker_aliases.get(ticker)
        if not alias or alias == ticker:
            continue

        alias_data = select_ticker_data(data, alias, ticker)
        if alias_data.empty:
            downloaded_alias = download_yahoo_request([alias], start, threads=False)
            alias_data = select_ticker_data(downloaded_alias, alias, ticker)

        methods[ticker]["yahoo_ticker"] = alias
        if not alias_data.empty:
            data = replace_ticker_data(data, alias_data, ticker)
            methods[ticker]["download_method"] = "alias"
        else:
            # Never keep data returned under a known reused/obsolete symbol.
            data = replace_ticker_data(data, pd.DataFrame(), ticker)
            methods[ticker]["download_method"] = "missing"

    # Some obsolete symbols have no safe continuous alias and Yahoo now maps
    # them to another security. Keep them in the historical universe report,
    # but never let the false price history enter factor calculations.
    for ticker in tickers:
        if ticker not in reused_tickers or ticker_aliases.get(ticker):
            continue

        data = replace_ticker_data(data, pd.DataFrame(), ticker)
        methods[ticker]["download_method"] = "reused_symbol_rejected"

    download_report = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "yahoo_ticker": methods[ticker]["yahoo_ticker"],
                "download_method": methods[ticker]["download_method"],
            }
            for ticker in tickers
        ]
    )

    method_counts = download_report["download_method"].value_counts()
    print(f"Recovered by individual retry: {method_counts.get('individual_retry', 0)}")
    print(f"Recovered by explicit alias: {method_counts.get('alias', 0)}")
    print(
        "Rejected reused Yahoo symbols: "
        f"{method_counts.get('reused_symbol_rejected', 0)}"
    )
    unavailable = (
        method_counts.get("missing", 0)
        + method_counts.get("reused_symbol_rejected", 0)
    )
    print(f"Tickers still unavailable: {unavailable}")

    if data.empty or not any(ticker_has_prices(data, ticker) for ticker in tickers):
        raise RuntimeError("yfinance returned no price data")

    return data.sort_index(), download_report
# -------------------------------------------------------------------------------------------------


# DATA CALCULATING
# -------------------------------------------------------------------------------------------------
# PRICES
def get_price_matrix(data):
    prices = data["Adj Close"].copy()
    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    return prices


# VOLUME
def get_volume_matrix(data):
    raw_volume = data["Volume"].copy().astype(float)
    raw_close = data["Close"].copy()
    adjusted_close = data["Adj Close"].copy()

    # Factor prices use Adj Close, while Yahoo volume is compatible with Close.
    # Rescale volume so Adj Close * volume equals Close * raw Yahoo volume.
    adjustment_ratio = raw_close.div(adjusted_close.where(adjusted_close.ne(0)))
    volume = raw_volume.mul(adjustment_ratio)
    volume = volume.sort_index()
    volume = volume.dropna(how="all")

    # Hide invalid data
    volume = volume.mask(volume < 0)
    return volume


# RETURNS
def compute_returns(prices, quality=None):
    returns = prices.pct_change(fill_method=None)

    if quality is not None:
        quality = quality.reindex(index=prices.index, columns=prices.columns).fillna(False)
        valid_return = quality & quality.shift(1, fill_value=False)
        excluded = (returns.notna() & ~valid_return).sum().sum()
        returns = returns.where(valid_return)
        print(f"Returns excluded by data quality: {excluded}")

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
        .dropna()
        .reset_index()
        .rename(columns={"level_1": "ticker", 0: "price"})
    )


# Forward Returns
def compute_forward_returns(prices, horizon=21, quality=None):
    fwd = prices.pct_change(horizon).shift(-horizon)

    if quality is not None:
        quality = quality.reindex(index=prices.index, columns=prices.columns).fillna(False)
        valid_forward_return = quality & quality.shift(-horizon, fill_value=False)
        fwd = fwd.where(valid_forward_return)

    return fwd


# Needed to created availability dataset for prices
def compute_availability(prices, membership, quality):
    return prices.notna() & membership & quality
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
def filter_universe(prices, liquidity, membership, quality, min_assets=150):
    initial_days = len(prices)

    valid_counts = compute_availability(prices, membership, quality).sum(axis=1)
    mask = valid_counts >= min_assets

    prices_filtered = prices.loc[mask]
    liquidity_filtered = liquidity.loc[mask]
    membership_filtered = membership.loc[mask]
    quality_filtered = quality.loc[mask]

    dropped_days = initial_days - len(prices_filtered)
    if dropped_days > 0:
        print(f"--- Universe Filter Applied ---")
        print(f"Dropped {dropped_days} days due to low asset count (min_assets={min_assets})")
        print(f"Remaining days: {len(prices_filtered)}")
    return prices_filtered, liquidity_filtered, membership_filtered, quality_filtered
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
    volume_quality,
    liquidity,
    prices_long,
    availability,
    quality,
    membership,
    forward_returns,
    universe_report,
):
    paths = {
        RAW_PRICES_PATH: prices,
        RETURNS_PATH: returns,
        FORWARD_RETURNS_PATH: forward_returns,
        VOLUME_PATH: volume,
        VOLUME_QUALITY_PATH: volume_quality,
        LIQUIDITY_PATH: liquidity,
        PRICES_LONG_PATH: prices_long,
        AVAILABILITY_PATH: availability,
        QUALITY_PATH: quality,
        MEMBERSHIP_PATH: membership,
    }

    for path, df in paths.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)

    # universe
    os.makedirs(os.path.dirname(UNIVERSE_PATH), exist_ok=True)
    universe_report.to_csv(UNIVERSE_PATH, index=False)
# -------------------------------------------------------------------------------------------------


def load_saved_equity_data():
    paths = (
        RAW_PRICES_PATH,
        RETURNS_PATH,
        VOLUME_PATH,
        LIQUIDITY_PATH,
        PRICES_LONG_PATH,
        AVAILABILITY_PATH,
        FORWARD_RETURNS_PATH,
    )
    return tuple(pd.read_parquet(path) for path in paths)




# -------------------------------------------------------------------------------------------------
# BUILD
def build_and_save_dataset(history, tickers):
    raw, download_report = download_data(tickers)

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

    quality, suspicious_return_mask, anomaly_trigger_mask, quarantined = (
        build_data_quality_mask(
            prices,
            SUSPICIOUS_ABS_DAILY_RETURN,
            MAX_ABS_DAILY_RETURN,
            ROUND_TRIP_RETURN_TOLERANCE,
            CONFIRMED_REAL_RETURN_EVENTS,
        )
    )

    # Full-period statistics are diagnostics only. They never delete past data.
    member_observations = membership.sum()
    available_member_observations = (prices.notna() & membership).sum()
    coverage = available_member_observations.div(member_observations).fillna(0)
    suspicious_daily_returns = (suspicious_return_mask & membership).sum()
    anomaly_triggers = (anomaly_trigger_mask & membership).sum()
    quarantine_date = first_true_date(quarantined)
    download_details = download_report.set_index("ticker").reindex(prices.columns)

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
        "yahoo_ticker": download_details["yahoo_ticker"].values,
        "download_method": download_details["download_method"].values,
        "first_membership_date": first_membership.reindex(prices.columns).values,
        "last_membership_date": last_membership.reindex(prices.columns).values,
        "membership_observations": member_observations.reindex(prices.columns).values,
        "price_coverage_during_membership": coverage.reindex(prices.columns).values,
        "has_price_during_membership": available_member_observations.gt(0)
        .reindex(prices.columns).values,
        "suspicious_daily_returns": suspicious_daily_returns.reindex(prices.columns).values,
        "anomaly_triggers": anomaly_triggers.reindex(prices.columns).values,
        "quarantined_from": quarantine_date.reindex(prices.columns).values,
        "has_any_price_data": prices.notna().any().reindex(prices.columns).values,
        "retained_in_dataset": True,
    })

    print(f"Historical tickers retained: {prices.shape[1]}")
    print(
        "Tickers with prices during membership: "
        f"{available_member_observations.gt(0).sum()}"
    )
    print(f"Point-in-time quarantined tickers: {quarantined.any().sum()}")

    returns = compute_returns(prices, quality)
    forward_returns = compute_forward_returns(prices, quality=quality)
    volume_quality = build_volume_quality_mask(volume)
    clean_volume = volume.where(volume_quality)
    liquidity = compute_liquidity(prices, clean_volume)

    # -------------------------
    # UNIVERSE FILTER (TIME LEVEL)
    # -------------------------
    prices, liquidity, membership, quality = filter_universe(
        prices,
        liquidity,
        membership,
        quality,
    )

    returns = returns.loc[prices.index]
    forward_returns = forward_returns.loc[prices.index]
    volume = volume.loc[prices.index]
    volume_quality = volume_quality.loc[prices.index]

    availability = compute_availability(prices, membership, quality)
    prices_long = to_long(prices)

    sanity_checks(prices, volume)
    check_extreme_gaps(prices, membership)
    print(returns.std().describe())
    print(forward_returns.std().describe())

    save_all(
        prices,
        returns,
        volume,
        volume_quality,
        liquidity,
        prices_long,
        availability,
        quality,
        membership,
        forward_returns,
        universe_report,
    )

    return prices, returns, volume, liquidity, prices_long, availability, forward_returns
# -------------------------------------------------------------------------------------------------
