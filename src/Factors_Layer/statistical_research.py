"""Robust statistical diagnostics for the three baseline factors.

This module does not select factor parameters and does not replace research.py.
It asks a narrower question: is the predictive relation of each baseline factor
to future cross-sectional returns statistically credible across horizons, time,
and monthly rebalance phases?
"""

import json
import os
import sys
import zlib

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factors import compute_momentum, compute_trend, compute_volatility
from pipeline import build_factor, load_data, load_membership


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR


# These are inference rules, not parameters selected after looking at results.
HORIZONS = (5, 21, 63, 126)
SIGNAL_LAG = 1
MIN_ASSETS = 30
CALENDAR_STEP = 21
ROLLING_WINDOW = 252
ROLLING_MIN_PERIODS = 126
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260904

PERIODS = {
    "full": ("2010-01-01", "2026-06-30"),
    "train": ("2010-01-01", "2017-12-31"),
    "validation": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-06-30"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "statistical_stage")


def build_baseline_factors(returns, prices, availability):
    """Build the current documented baseline specifications once."""
    momentum = build_factor(
        compute_momentum(returns, window=252, skip=21, min_obs=200),
        availability,
    )
    low_vol = -build_factor(
        compute_volatility(returns, window=60, min_obs=40),
        availability,
    )
    trend = build_factor(
        compute_trend(prices, window=50, min_obs=10),
        availability,
    )

    return {
        "momentum_12m_1m": momentum,
        "low_vol_60d": low_vol,
        "trend_sma50": trend,
    }


def compute_forward_returns(prices, horizon):
    """Return adjusted-price returns from t to t+h without implicit filling."""
    return prices.pct_change(horizon, fill_method=None).shift(-horizon)


def compute_daily_spearman_ic(
    factor,
    forward_returns,
    membership,
    signal_lag=SIGNAL_LAG,
    min_assets=MIN_ASSETS,
):
    """Calculate a daily cross-sectional Spearman IC and valid asset count."""
    scores = factor.shift(signal_lag)
    scores, future_returns = scores.align(forward_returns, join="inner")
    membership = membership.reindex(index=scores.index, columns=scores.columns)

    valid = scores.notna() & future_returns.notna() & membership.fillna(False)
    asset_count = valid.sum(axis=1).astype("int32")

    score_ranks = scores.where(valid).rank(axis=1, method="average")
    return_ranks = future_returns.where(valid).rank(axis=1, method="average")
    score_centered = score_ranks.sub(score_ranks.mean(axis=1), axis=0)
    return_centered = return_ranks.sub(return_ranks.mean(axis=1), axis=0)

    numerator = (score_centered * return_centered).sum(axis=1, min_count=1)
    denominator = np.sqrt(
        score_centered.pow(2).sum(axis=1, min_count=1)
        * return_centered.pow(2).sum(axis=1, min_count=1)
    )
    ic = (numerator / denominator).where(asset_count >= min_assets)
    ic = ic.replace([np.inf, -np.inf], np.nan).rename("ic")

    return pd.DataFrame({"ic": ic, "asset_count": asset_count})


def hac_mean_test(values, maxlags):
    """Estimate the mean and its Newey-West/HAC uncertainty."""
    values = pd.Series(values).dropna().astype(float)

    if len(values) < 3:
        return np.nan, np.nan, np.nan

    effective_lags = min(int(maxlags), len(values) - 1)
    model = sm.OLS(values.to_numpy(), np.ones((len(values), 1)))
    fitted = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": effective_lags, "use_correction": True},
    )

    return float(fitted.bse[0]), float(fitted.tvalues[0]), float(fitted.pvalues[0])


def circular_block_bootstrap_ci(values, block_size, samples, seed):
    """Bootstrap the mean while preserving local serial dependence."""
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    observation_count = len(values)

    if observation_count < 3:
        return np.nan, np.nan

    block_size = min(int(block_size), observation_count)
    blocks_per_sample = int(np.ceil(observation_count / block_size))
    offsets = np.arange(block_size)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    batch_size = 200

    for first in range(0, samples, batch_size):
        current_batch = min(batch_size, samples - first)
        starts = rng.integers(
            0,
            observation_count,
            size=(current_batch, blocks_per_sample),
        )
        indices = (starts[..., None] + offsets) % observation_count
        indices = indices.reshape(current_batch, -1)[:, :observation_count]
        means[first : first + current_batch] = values[indices].mean(axis=1)

    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def stable_seed(*parts):
    """Make every bootstrap result reproducible across Python sessions."""
    key = "|".join(map(str, parts)).encode("utf-8")
    return (BOOTSTRAP_SEED + zlib.crc32(key)) % (2**32)


def summarize_ic(ic_data, factor_name, horizon, period_name, start_date, end_date):
    """Summarize one factor, horizon, and predeclared time period."""
    period_data = ic_data.loc[start_date:end_date]
    valid_ic = period_data["ic"].dropna()
    matching_counts = period_data.loc[valid_ic.index, "asset_count"]
    observations = len(valid_ic)
    mean_ic = valid_ic.mean() if observations else np.nan
    std_ic = valid_ic.std() if observations else np.nan
    iid_tstat = (
        mean_ic / std_ic * np.sqrt(observations)
        if observations > 1 and std_ic != 0
        else np.nan
    )
    nw_se, nw_tstat, nw_pvalue = hac_mean_test(valid_ic, horizon - 1)
    bootstrap_low, bootstrap_high = circular_block_bootstrap_ci(
        valid_ic,
        block_size=horizon,
        samples=BOOTSTRAP_SAMPLES,
        seed=stable_seed(factor_name, horizon, period_name),
    )

    return {
        "period": period_name,
        "factor": factor_name,
        "horizon_days": horizon,
        "observations": observations,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "iid_tstat": iid_tstat,
        "nw_se": nw_se,
        "nw_tstat": nw_tstat,
        "nw_pvalue": nw_pvalue,
        "bootstrap_95_low": bootstrap_low,
        "bootstrap_95_high": bootstrap_high,
        "positive_rate": (valid_ic > 0).mean() if observations else np.nan,
        "average_assets": matching_counts.mean() if observations else np.nan,
        "minimum_assets": matching_counts.min() if observations else np.nan,
        "ic_autocorr_1": valid_ic.autocorr(1),
        "ic_autocorr_horizon": valid_ic.autocorr(horizon),
    }


def summarize_calendar_offsets(
    ic_data,
    factor_name,
    horizon,
    period_name,
    start_date,
    end_date,
):
    """Measure sensitivity to all 21 possible monthly rebalance phases."""
    period_data = ic_data.loc[start_date:end_date]
    rows = []
    offset_hac_lags = max(0, int(np.ceil(horizon / CALENDAR_STEP)) - 1)

    for offset in range(CALENDAR_STEP):
        sampled = period_data.iloc[offset::CALENDAR_STEP]
        valid_ic = sampled["ic"].dropna()
        counts = sampled.loc[valid_ic.index, "asset_count"]
        nw_se, nw_tstat, nw_pvalue = hac_mean_test(valid_ic, offset_hac_lags)

        rows.append(
            {
                "period": period_name,
                "factor": factor_name,
                "horizon_days": horizon,
                "offset": offset,
                "first_date": valid_ic.index.min() if not valid_ic.empty else pd.NaT,
                "observations": len(valid_ic),
                "mean_ic": valid_ic.mean() if not valid_ic.empty else np.nan,
                "positive_rate": (
                    (valid_ic > 0).mean() if not valid_ic.empty else np.nan
                ),
                "nw_se": nw_se,
                "nw_tstat": nw_tstat,
                "nw_pvalue": nw_pvalue,
                "average_assets": counts.mean() if not counts.empty else np.nan,
            }
        )

    return rows


def add_multiple_testing_corrections(statistics):
    """Correct the 12 baseline factor-horizon tests within each period."""
    statistics = statistics.copy()
    statistics["nw_pvalue_holm"] = np.nan
    statistics["nw_pvalue_fdr_bh"] = np.nan

    for _, group in statistics.groupby("period", sort=False):
        valid = group["nw_pvalue"].notna()
        index = group.index[valid]

        if index.empty:
            continue

        pvalues = statistics.loc[index, "nw_pvalue"].to_numpy()
        statistics.loc[index, "nw_pvalue_holm"] = multipletests(
            pvalues,
            method="holm",
        )[1]
        statistics.loc[index, "nw_pvalue_fdr_bh"] = multipletests(
            pvalues,
            method="fdr_bh",
        )[1]

    return statistics


def run_statistical_research():
    """Run and save the complete first-stage statistical diagnosis."""
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    factors = build_baseline_factors(returns, prices, availability)

    daily_columns = {}
    rolling_columns = {}
    statistic_rows = []
    offset_rows = []

    for horizon in HORIZONS:
        forward_returns = compute_forward_returns(prices, horizon)

        for factor_name, factor in factors.items():
            ic_data = compute_daily_spearman_ic(
                factor,
                forward_returns,
                membership,
            )
            key = f"{factor_name}__h{horizon}"
            daily_columns[f"{key}__ic"] = ic_data["ic"]
            daily_columns[f"{key}__assets"] = ic_data["asset_count"]
            rolling_columns[key] = ic_data["ic"].rolling(
                ROLLING_WINDOW,
                min_periods=ROLLING_MIN_PERIODS,
            ).mean()

            for period_name, (start_date, end_date) in PERIODS.items():
                statistic_rows.append(
                    summarize_ic(
                        ic_data,
                        factor_name,
                        horizon,
                        period_name,
                        start_date,
                        end_date,
                    )
                )
                offset_rows.extend(
                    summarize_calendar_offsets(
                        ic_data,
                        factor_name,
                        horizon,
                        period_name,
                        start_date,
                        end_date,
                    )
                )

    daily_ic = pd.DataFrame(daily_columns)
    rolling_ic = pd.DataFrame(rolling_columns)
    statistics = add_multiple_testing_corrections(pd.DataFrame(statistic_rows))
    calendar_offsets = pd.DataFrame(offset_rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily_ic.to_parquet(os.path.join(OUTPUT_DIR, "daily_ic_and_asset_counts.parquet"))
    rolling_ic.to_parquet(os.path.join(OUTPUT_DIR, "rolling_252d_mean_ic.parquet"))
    statistics.to_csv(os.path.join(OUTPUT_DIR, "ic_statistics.csv"), index=False)
    calendar_offsets.to_csv(
        os.path.join(OUTPUT_DIR, "calendar_offset_statistics.csv"),
        index=False,
    )
    run_config = {
        "data_start": str(prices.index.min().date()),
        "data_end": str(prices.index.max().date()),
        "assets_in_price_matrix": int(prices.shape[1]),
        "horizons_trading_days": list(HORIZONS),
        "signal_lag_trading_days": SIGNAL_LAG,
        "minimum_assets_per_ic": MIN_ASSETS,
        "calendar_offsets": CALENDAR_STEP,
        "rolling_window_trading_days": ROLLING_WINDOW,
        "rolling_minimum_observations": ROLLING_MIN_PERIODS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_block_size": "equal to forward horizon",
        "daily_hac_lags": "forward horizon minus 1",
        "multiple_testing_family": "3 factors x 4 horizons within each period",
        "periods": PERIODS,
    }
    with open(
        os.path.join(OUTPUT_DIR, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_config, file, indent=2)
    plot_rolling_ic(rolling_ic)
    plot_calendar_offsets(calendar_offsets)

    return daily_ic, rolling_ic, statistics, calendar_offsets


def plot_rolling_ic(rolling_ic):
    """Save rolling one-year Mean IC paths for all factors and horizons."""
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)

    for axis, horizon in zip(axes.flat, HORIZONS):
        for column in rolling_ic.columns:
            if column.endswith(f"__h{horizon}"):
                label = column.rsplit("__h", maxsplit=1)[0]
                axis.plot(rolling_ic.index, rolling_ic[column], label=label)

        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_title(f"Forward horizon: {horizon} trading days")
        axis.set_ylabel("Rolling 252-day Mean IC")
        axis.grid(alpha=0.25)

    axes[0, 0].legend()
    figure.suptitle("Baseline factors: IC stability through time")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "rolling_ic.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_calendar_offsets(calendar_offsets):
    """Save full-period IC dependence on the monthly rebalance phase."""
    full = calendar_offsets[calendar_offsets["period"] == "full"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)

    for axis, horizon in zip(axes.flat, HORIZONS):
        horizon_data = full[full["horizon_days"] == horizon]

        for factor_name, factor_data in horizon_data.groupby("factor"):
            axis.plot(
                factor_data["offset"],
                factor_data["mean_ic"],
                marker="o",
                markersize=3,
                label=factor_name,
            )

        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_title(f"Forward horizon: {horizon} trading days")
        axis.set_xlabel("Monthly phase offset (0-20)")
        axis.set_ylabel("Mean IC")
        axis.set_xticks(range(0, CALENDAR_STEP, 2))
        axis.grid(alpha=0.25)

    axes[0, 0].legend()
    figure.suptitle("Full-period sensitivity to monthly rebalance date")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "calendar_offset_sensitivity.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_results(statistics):
    """Print the main robust statistics without hiding failed hypotheses."""
    columns = [
        "period",
        "factor",
        "horizon_days",
        "observations",
        "mean_ic",
        "nw_tstat",
        "nw_pvalue",
        "nw_pvalue_holm",
        "nw_pvalue_fdr_bh",
        "bootstrap_95_low",
        "bootstrap_95_high",
        "positive_rate",
        "average_assets",
    ]
    display = statistics[columns].sort_values(
        ["period", "factor", "horizon_days"]
    )
    print("\nROBUST BASELINE FACTOR STATISTICS")
    print("HAC lag = horizon - 1; corrections cover 12 tests per period.")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    _, _, statistics, _ = run_statistical_research()
    print_results(statistics)
