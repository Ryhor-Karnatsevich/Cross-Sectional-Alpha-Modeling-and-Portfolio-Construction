import pandas as pd
import numpy as np
import sys
import os
from transforms import zscore, winsorize
from factors import compute_momentum, compute_volatility


# ------------------------------------------------------------------------------------------------
# Get data from config.py
data_system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_System'))
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)
from config import RETURNS_PATH, AVAILABILITY_PATH, FORWARD_RETURNS_PATH
# ------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# LOAD DATA from parquets
def load_data():
    returns = pd.read_parquet(RETURNS_PATH)
    availability = pd.read_parquet(AVAILABILITY_PATH)
    forward_returns = pd.read_parquet(FORWARD_RETURNS_PATH)
    # Join tables
    returns, forward_returns = returns.align(forward_returns, join="inner")
    availability = availability.loc[returns.index, returns.columns]

    return returns, availability, forward_returns
# -----------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# BUILD FACTORS
# 1.Momentum
def build_factors(returns, availability):
    momentum = compute_momentum(returns)
    # MASK
    momentum = momentum.where(availability)
    # WINSORIZE
    momentum = winsorize(momentum)
    # Z-SCORE
    momentum = zscore(momentum)

    return momentum


# 2.Volatility
def build_low_vol_factor(returns, availability):
    vol = compute_volatility(returns)

    # Mask
    vol = vol.where(availability)
    # Winsorize
    vol = winsorize(vol)
    # Z-score
    vol = zscore(vol)

    #
    vol = -vol

    return vol
# ----------------------------------------------------------------------------------------------------------------------

def print_ic(factor, name, forward_returns):
    ic = compute_ic(factor, forward_returns)

    print(f"\n{'=' * 50}")
    print(f"{name} IC stats:")
    print(f"{'=' * 50}")
    print(f"Mean IC: {ic.mean():.6f}")
    print(f"Std IC: {ic.std():.6f}")
    print(f"T-stat: {ic.mean() / ic.std() * np.sqrt(ic.notna().sum()):.4f}")
    print(f"\nIC distribution:")
    print(ic.describe())
    print(f"\nIC > 0: {(ic > 0).mean():.2%} of days")

    return ic
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

    # Low Volatility
    low_vol = build_low_vol_factor(returns, availability)
    ic_vol = print_ic(low_vol, "Low Volatility", forward_returns)

    return momentum, low_vol, ic, ic_vol


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    momentum, low_vol, ic_mom, ic_vol = run_factor_pipeline()

    print("\n" + "=" * 50)
    print("FINAL COMPARISON:")
    print("=" * 50)
    print(f"Momentum IC:     {ic_mom.mean():.6f}")
    print(f"Low Volatility IC: {ic_vol.mean():.6f}")