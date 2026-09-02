import numpy as np


# -------------------------
# MOMENTUM (12-1 style)
def compute_momentum(returns, window=252, skip=21, min_obs=200):
    formation_window = window - skip

    if formation_window <= 0:
        raise ValueError("Momentum window must be greater than skip")
    if not 1 <= min_obs <= formation_window:
        raise ValueError("min_obs must be between 1 and window - skip")

    log_ret = np.log1p(returns)
    formation_returns = log_ret.shift(skip)

    return formation_returns.rolling(
        formation_window,
        min_periods=min_obs,
    ).sum()


# -------------------------
# LOW VOLATILITY
def compute_volatility(returns, window=60, min_obs=40):
    if not 2 <= min_obs <= window:
        raise ValueError("min_obs must be between 2 and window")

    return returns.rolling(window, min_periods=min_obs).std()


# -------------------------
# TREND (PRICE / SMA - 1)
def compute_trend(prices, window=50, min_obs=10):
    if not 1 <= min_obs <= window:
        raise ValueError("min_obs must be between 1 and window")

    sma = prices.rolling(window, min_periods=min_obs).mean()

    return prices / sma - 1

