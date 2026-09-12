import numpy as np
import pandas as pd


# -------------------------
# MOMENTUM (12-1 style)
def compute_momentum(returns, window, skip, min_obs):
    return _cumulative_log_return(returns, window, skip, min_obs)


# -------------------------
# LOW VOLATILITY
def compute_low_volatility(returns, window, min_obs):
    if not 2 <= min_obs <= window:
        raise ValueError("min_obs must be between 2 and window")

    return -returns.rolling(window, min_periods=min_obs).std()


# -------------------------
# TREND (PRICE / SMA - 1)
def compute_trend(prices, window, min_obs):
    if not 1 <= min_obs <= window:
        raise ValueError("min_obs must be between 1 and window")

    sma = prices.rolling(window, min_periods=min_obs).mean()

    return prices / sma - 1


# -------------------------
# SHORT-TERM REVERSAL
def compute_short_term_reversal(returns, window, min_obs):
    return -_cumulative_log_return(returns, window, 0, min_obs)


# -------------------------
# RESIDUAL MOMENTUM
def compute_residual_momentum(
    returns,
    availability,
    window,
    skip,
    min_obs,
):
    eligible_returns = returns.where(availability)
    market_return = eligible_returns.mean(axis=1)
    residual_returns = eligible_returns.sub(market_return, axis=0)

    return _cumulative_log_return(residual_returns, window, skip, min_obs)


# -------------------------
# VOLATILITY-SCALED MOMENTUM
def compute_volatility_scaled_momentum(
    returns,
    window,
    skip,
    momentum_min_obs,
    volatility_window,
    volatility_min_obs,
):
    momentum = _cumulative_log_return(
        returns,
        window,
        skip,
        momentum_min_obs,
    )
    volatility = returns.rolling(
        volatility_window,
        min_periods=volatility_min_obs,
    ).std()

    return momentum / volatility.replace(0, np.nan)


# -------------------------
# HIGH PROXIMITY
def compute_high_proximity(prices, window, min_obs):
    if not 1 <= min_obs <= window:
        raise ValueError("min_obs must be between 1 and window")

    rolling_high = prices.rolling(window, min_periods=min_obs).max()

    return prices / rolling_high - 1


# -------------------------
# TREND SLOPE
def compute_trend_slope(prices, window, min_obs, annualization_factor):
    return _rolling_log_price_slope(
        prices,
        window,
        min_obs,
        annualization_factor,
    )


# -------------------------
# RISK-ADJUSTED TREND
def compute_risk_adjusted_trend(
    prices,
    returns,
    window,
    min_obs,
    annualization_factor,
):
    slope = _rolling_log_price_slope(
        prices,
        window,
        min_obs,
        annualization_factor,
    )
    volatility = (
        returns.rolling(window, min_periods=min_obs).std()
        * np.sqrt(annualization_factor)
    )

    return slope / volatility.replace(0, np.nan)


# -------------------------
# LIQUIDITY CHANGE
def compute_liquidity_change(
    prices,
    volume,
    short_window,
    long_window,
    short_min_obs,
    long_min_obs,
):
    dollar_volume = prices * volume
    short_average = dollar_volume.rolling(
        short_window,
        min_periods=short_min_obs,
    ).mean()
    long_average = dollar_volume.rolling(
        long_window,
        min_periods=long_min_obs,
    ).mean()
    valid = (short_average > 0) & (long_average > 0)
    ratio = (short_average / long_average).where(valid)

    return np.log(ratio)


# -------------------------
# PRICE-VOLUME CONFIRMATION
def compute_price_volume_confirmation(
    momentum,
    liquidity_change,
    confirmation_strength,
    liquidity_clip,
):
    if liquidity_clip <= 0:
        raise ValueError("liquidity_clip must be positive")

    bounded_liquidity = liquidity_change.clip(-liquidity_clip, liquidity_clip)

    return momentum * (1 + confirmation_strength * bounded_liquidity)


# -------------------------
# INTERNAL HELPERS
# Converts daily returns to log returns, skips the most recent observations
# and sums the remaining returns inside the rolling formation window.
def _cumulative_log_return(returns, window, skip, min_obs):
    formation_window = window - skip

    if formation_window <= 0:
        raise ValueError("window must be greater than skip")
    if not 1 <= min_obs <= formation_window:
        raise ValueError("min_obs must be between 1 and window - skip")

    log_ret = np.log1p(returns)
    formation_returns = log_ret.shift(skip)

    return formation_returns.rolling(
        formation_window,
        min_periods=min_obs,
    ).sum()


# Fits a linear trend to log prices inside every rolling window.
# Handles missing prices and converts the estimated daily slope to an annual rate.
def _rolling_log_price_slope(prices, window, min_obs, annualization_factor):
    if not 2 <= min_obs <= window:
        raise ValueError("min_obs must be between 2 and window")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    log_prices = np.log(prices)
    observation_number = pd.Series(
        np.arange(len(prices), dtype=float),
        index=prices.index,
    )
    observed = log_prices.notna().astype(float)
    x = observed.mul(observation_number, axis=0)
    x_squared = observed.mul(observation_number.pow(2), axis=0)
    xy = log_prices.mul(observation_number, axis=0)

    count = observed.rolling(window, min_periods=min_obs).sum()
    sum_x = x.rolling(window, min_periods=min_obs).sum()
    sum_y = log_prices.rolling(window, min_periods=min_obs).sum()
    sum_x_squared = x_squared.rolling(window, min_periods=min_obs).sum()
    sum_xy = xy.rolling(window, min_periods=min_obs).sum()
    denominator = count * sum_x_squared - sum_x.pow(2)
    slope = (count * sum_xy - sum_x * sum_y) / denominator

    return (slope * annualization_factor).where(denominator > 0)
