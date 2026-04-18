import numpy as np


def compute_momentum(returns, window=252, min_obs=200):
    log_ret = np.log1p(returns)

    mom = log_ret.rolling(window).sum()

    valid_obs = returns.notna().rolling(window).sum()
    mom = mom.where(valid_obs >= min_obs)

    return mom

