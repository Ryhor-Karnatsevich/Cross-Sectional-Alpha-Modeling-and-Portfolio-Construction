import pandas as pd
import numpy as np
import sys
import os

# -------------------------
# PATH SETUP
# -------------------------
data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Data_System')
)

if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import RETURNS_PATH, AVAILABILITY_PATH, FORWARD_RETURNS_PATH


# -------------------------
# LOAD DATA
# -------------------------
def load_data():
    returns = pd.read_parquet(RETURNS_PATH)
    availability = pd.read_parquet(AVAILABILITY_PATH)
    forward_returns = pd.read_parquet(FORWARD_RETURNS_PATH)

    # ЖЁСТКОЕ выравнивание (очень важно)
    returns, forward_returns = returns.align(forward_returns, join="inner")
    availability = availability.loc[returns.index, returns.columns]

    return returns, availability, forward_returns


# -------------------------
# TRANSFORMS
# -------------------------
def winsorize(df, lower=0.01, upper=0.99):
    lower_q = df.quantile(lower, axis=1)
    upper_q = df.quantile(upper, axis=1)

    return df.clip(lower=lower_q, upper=upper_q, axis=0)


def zscore(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    z = df.sub(mean, axis=0).div(std, axis=0)

    return z


# -------------------------
# FACTOR: MOMENTUM
# -------------------------
def compute_momentum(returns, window=252, skip=21, min_obs=200):
    log_ret = np.log1p(returns)

    mom = log_ret.rolling(252).sum()
    mom = mom - log_ret.rolling(21).sum()

    valid_obs = returns.notna().rolling(window).sum()
    mom = mom.where(valid_obs >= min_obs)

    return mom




# -------------------------
# BUILD FACTORS
# -------------------------
def build_factors(returns, availability):
    momentum = compute_momentum(returns)

    # MASK
    momentum = momentum.where(availability)

    # WINSORIZE
    momentum = winsorize(momentum)

    # Z-SCORE
    momentum = zscore(momentum)

    return momentum


# -------------------------
# IC COMPUTATION
# -------------------------
def compute_ic(factor, forward_returns):
    ic_list = []

    for date in factor.index:
        x = factor.loc[date]
        y = forward_returns.loc[date]

        mask = x.notna() & y.notna()

        if mask.sum() < 20:
            ic_list.append(np.nan)
            continue

        ic = x[mask].corr(y[mask], method="spearman")
        #ic = ic.rolling(63).mean()
        ic_list.append(ic)


    return pd.Series(ic_list, index=factor.index)


# -------------------------
# PIPELINE
# -------------------------
def run_factor_pipeline():
    returns, availability, forward_returns = load_data()

    momentum = build_factors(returns, availability)

    # -------------------------
    # DIAGNOSTICS
    # -------------------------
    print("\nMomentum diagnostics:")
    print("Mean (should be ~0):", momentum.mean(axis=1).mean())
    print("Std (should be ~1):", momentum.std(axis=1).mean())

    print("\nCross-sectional sample:")
    print(momentum.iloc[300].describe())

    print("\nCoverage:")
    print(momentum.notna().sum(axis=1).describe())

    # -------------------------
    # IC
    # -------------------------
    ic = compute_ic(momentum, forward_returns)

    print("\nIC stats:")
    print("Mean IC:", ic.mean())
    print("Std IC:", ic.std())
    print("T-stat:", ic.mean() / ic.std() * np.sqrt(ic.notna().sum()))

    print("\nIC distribution:")
    print(ic.describe())

    return momentum, ic


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    momentum, ic = run_factor_pipeline()