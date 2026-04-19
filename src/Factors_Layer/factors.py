import numpy as np


# -------------------------
# MOMENTUM (12-1 style)
def compute_momentum(returns, window=252, skip=21, min_obs=200):
    log_ret = np.log1p(returns)

    long = log_ret.rolling(window).sum()
    short = log_ret.rolling(skip).sum()

    # CRITICAL: no shift here
    mom = long - short

    valid_obs = returns.notna().rolling(window).sum()
    mom = mom.where(valid_obs >= min_obs)

    return mom


# -------------------------
# LOW VOLATILITY
def compute_volatility(returns, window=60, min_obs=40):
    vol = returns.rolling(window).std()

    valid_obs = returns.notna().rolling(window).sum()
    vol = vol.where(valid_obs >= min_obs)

    return vol


# -------------------------
# TREND (PRICE / SMA - 1)
def compute_trend(prices, window=50, min_obs=10):
    sma = prices.rolling(window).mean()

    trend = prices / sma - 1

    valid_obs = prices.notna().rolling(window).sum()
    trend = trend.where(valid_obs >= min_obs)

    return trend

