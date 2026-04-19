import pandas as pd
import numpy as np
import sys
import os

from transforms import zscore, winsorize
from factors import compute_momentum, compute_volatility, compute_trend

# config
data_system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_System'))
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import RETURNS_PATH, AVAILABILITY_PATH, FORWARD_RETURNS_PATH, RAW_PRICES_PATH


# -------------------------
# LOAD
# -------------------------
def load_data():
    returns = pd.read_parquet(RETURNS_PATH)
    availability = pd.read_parquet(AVAILABILITY_PATH)
    forward_returns = pd.read_parquet(FORWARD_RETURNS_PATH)
    prices = pd.read_parquet(RAW_PRICES_PATH)

    returns, forward_returns = returns.align(forward_returns, join="inner")
    availability = availability.loc[returns.index, returns.columns]
    prices = prices.loc[returns.index, returns.columns]

    return returns, availability, forward_returns, prices


# -------------------------
# BUILD FACTORS
# -------------------------
def build_factor(raw, availability):
    raw = raw.where(availability)
    raw = winsorize(raw)
    raw = zscore(raw)
    return raw


# -------------------------
# IC
# -------------------------
def compute_ic(factor, forward_returns):
    ic_list = []

    for date in factor.index:
        x = factor.loc[date]
        y = forward_returns.loc[date]

        mask = x.notna() & y.notna()

        if mask.sum() < 30:
            ic_list.append(np.nan)
            continue

        ic = x[mask].corr(y[mask], method="spearman")
        ic = ic.iloc[::21]
        ic_list.append(ic)

    return pd.Series(ic_list, index=factor.index)


# -------------------------
# IC PRINT
# -------------------------
def print_ic(name, ic):
    print(f"\n{'='*50}")
    print(f"{name} IC stats")
    print(f"{'='*50}")

    mean = ic.mean()
    std = ic.std()
    tstat = mean / std * np.sqrt(ic.notna().sum())

    print(f"Mean IC: {mean:.6f}")
    print(f"Std IC: {std:.6f}")
    print(f"T-stat: {tstat:.4f}")
    print(f"IC > 0: {(ic > 0).mean():.2%}")

    print("\nIC autocorr:")
    print("lag1:", ic.autocorr(1))
    print("lag5:", ic.autocorr(5))


# -------------------------
# PIPELINE
# -------------------------
def run_pipeline():
    returns, availability, forward_returns, prices = load_data()

    # RAW factors
    mom_raw = compute_momentum(returns)
    vol_raw = compute_volatility(returns)
    trend_raw = compute_trend(prices)

    # TRANSFORM
    momentum = build_factor(mom_raw, availability)
    low_vol = -build_factor(vol_raw, availability)
    trend = build_factor(trend_raw, availability)

    # IC
    ic_mom = compute_ic(momentum, forward_returns)
    ic_vol = compute_ic(low_vol, forward_returns)
    ic_trend = compute_ic(trend, forward_returns)

    # PRINT
    print_ic("Momentum", ic_mom)
    print_ic("Low Vol", ic_vol)
    print_ic("Trend", ic_trend)

    print("\n" + "="*60)
    print("FACTOR COMPARISON")
    print("="*60)
    print(f"Momentum:   {ic_mom.mean():.6f}")
    print(f"Low Vol:    {ic_vol.mean():.6f}")
    print(f"Trend:      {ic_trend.mean():.6f}")

    return momentum, low_vol, trend, ic_mom, ic_vol, ic_trend


if __name__ == "__main__":
    run_pipeline()