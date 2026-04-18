import numpy as np


def compute_momentum(returns, window=252, skip=21, min_obs=200):
    log_ret = np.log1p(returns)

    mom = log_ret.rolling(252).sum()
    mom = mom - log_ret.rolling(skip).sum()

    valid_obs = returns.notna().rolling(window).sum()
    mom = mom.where(valid_obs >= min_obs)

    return mom



def compute_volatility(returns, window=60, min_obs=40):
    vol = returns.rolling(window).std()

    # minimum amount of observations
    valid_obs = returns.notna().rolling(window).sum()
    vol = vol.where(valid_obs >= min_obs)

    return vol

