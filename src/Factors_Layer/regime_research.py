"""Stage 6: point-in-time market-regime diagnostics.

Regimes are defined only with information available before the IC evaluation
date. They diagnose instability; they are not used to retrofit trading rules.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factor_independence import build_baseline_factors, load_selected_factors
from pipeline import load_data, load_membership
from statistical_research import PERIODS, compute_daily_spearman_ic, compute_forward_returns
from walk_forward import OUTPUT_DIR as WALK_FORWARD_OUTPUT_DIR


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR


OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "regime_stage")

REGIME_DEFINITIONS = {
    "market_trend": ("bear", "bull"),
    "market_volatility": ("low_vol", "high_vol"),
    "market_breadth": ("narrow", "broad"),
}

OOS_PERIODS = {
    "oos_full": ("2015-01-01", "2026-06-30"),
    "oos_early": ("2015-01-01", "2021-12-31"),
    "oos_late": ("2022-01-01", "2026-06-30"),
}


def lagged_boolean(current_value, threshold):
    current_value = current_value.shift(1)
    threshold = threshold.shift(1)
    result = pd.Series(pd.NA, index=current_value.index, dtype="boolean")
    valid = current_value.notna() & threshold.notna()
    result.loc[valid] = current_value.loc[valid] > threshold.loc[valid]
    return result


def build_point_in_time_regimes(returns, prices, availability):
    """Create bull/bear, volatility, and breadth states known at t-1."""
    eligible_returns = returns.where(availability)
    market_return = eligible_returns.mean(axis=1)
    market_index = (1 + market_return.fillna(0)).cumprod()
    market_average = market_index.rolling(200, min_periods=160).mean()

    market_volatility = market_return.rolling(63, min_periods=50).std() * np.sqrt(252)
    historical_volatility_threshold = (
        market_volatility.expanding(min_periods=252).median()
    )

    stock_average = prices.rolling(200, min_periods=160).mean()
    breadth_numerator = ((prices > stock_average) & availability).sum(axis=1)
    breadth_denominator = availability.sum(axis=1).replace(0, np.nan)
    breadth = breadth_numerator / breadth_denominator
    historical_breadth_threshold = breadth.expanding(min_periods=252).median()

    return pd.DataFrame(
        {
            "market_trend": lagged_boolean(market_index, market_average),
            "market_volatility": lagged_boolean(
                market_volatility,
                historical_volatility_threshold,
            ),
            "market_breadth": lagged_boolean(
                breadth,
                historical_breadth_threshold,
            ),
            "market_index_t_minus_1": market_index.shift(1),
            "market_sma200_t_minus_1": market_average.shift(1),
            "market_volatility_t_minus_1": market_volatility.shift(1),
            "volatility_threshold_t_minus_1": historical_volatility_threshold.shift(1),
            "breadth_t_minus_1": breadth.shift(1),
            "breadth_threshold_t_minus_1": historical_breadth_threshold.shift(1),
        }
    )


def regime_difference_test(ic, state, maxlags):
    """Estimate state-1 minus state-0 Mean IC with chronological HAC OLS."""
    data = pd.concat([ic.rename("ic"), state.rename("state")], axis=1).dropna()

    if len(data) < 30 or data["state"].nunique() < 2:
        return {
            "observations": len(data),
            "state_0_observations": np.nan,
            "state_1_observations": np.nan,
            "state_0_mean_ic": np.nan,
            "state_1_mean_ic": np.nan,
            "difference_state_1_minus_0": np.nan,
            "difference_hac_tstat": np.nan,
            "difference_pvalue": np.nan,
        }

    state_values = data["state"].astype(int)
    design = sm.add_constant(state_values.to_numpy(dtype=float))
    fitted = sm.OLS(data["ic"].to_numpy(dtype=float), design).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": min(int(maxlags), len(data) - 1),
            "use_correction": True,
        },
    )
    state_0 = data.loc[~data["state"].astype(bool), "ic"]
    state_1 = data.loc[data["state"].astype(bool), "ic"]

    return {
        "observations": len(data),
        "state_0_observations": len(state_0),
        "state_1_observations": len(state_1),
        "state_0_mean_ic": state_0.mean(),
        "state_1_mean_ic": state_1.mean(),
        "difference_state_1_minus_0": fitted.params[1],
        "difference_hac_tstat": fitted.tvalues[1],
        "difference_pvalue": fitted.pvalues[1],
    }


def add_regime_multiple_testing(results):
    results = results.copy()
    results["difference_pvalue_holm"] = np.nan
    results["difference_pvalue_fdr_bh"] = np.nan

    for _, group in results.groupby(["analysis", "period"]):
        valid = group["difference_pvalue"].notna()
        index = group.index[valid]
        if index.empty:
            continue
        pvalues = results.loc[index, "difference_pvalue"].to_numpy()
        results.loc[index, "difference_pvalue_holm"] = multipletests(
            pvalues,
            method="holm",
        )[1]
        results.loc[index, "difference_pvalue_fdr_bh"] = multipletests(
            pvalues,
            method="fdr_bh",
        )[1]

    return results


def fixed_factor_regime_tests(
    selected,
    selected_factors,
    baselines,
    prices,
    membership,
    regimes,
):
    factors = {**baselines, **selected_factors}
    horizons = {
        "baseline_momentum": 21,
        "baseline_low_vol": 21,
        "baseline_trend": 21,
    }
    for row in selected.itertuples():
        horizons[f"{row.family}__{row.variant}"] = int(row.horizon_days)

    rows = []
    forward_returns = {
        horizon: compute_forward_returns(prices, horizon)
        for horizon in sorted(set(horizons.values()))
    }

    for factor_name, factor in factors.items():
        horizon = horizons[factor_name]
        ic = compute_daily_spearman_ic(
            factor,
            forward_returns[horizon],
            membership,
        )["ic"]

        for period_name, (start_date, end_date) in PERIODS.items():
            period_ic = ic.loc[start_date:end_date]

            for regime_name, state_labels in REGIME_DEFINITIONS.items():
                result = regime_difference_test(
                    period_ic,
                    regimes[regime_name].loc[start_date:end_date],
                    horizon - 1,
                )
                rows.append(
                    {
                        "analysis": "fixed_factor",
                        "period": period_name,
                        "factor": factor_name,
                        "horizon_days": horizon,
                        "regime": regime_name,
                        "state_0_label": state_labels[0],
                        "state_1_label": state_labels[1],
                        **result,
                    }
                )

    return pd.DataFrame(rows)


def walk_forward_regime_tests(oos_ic, regimes):
    rows = []

    for family, family_data in oos_ic.groupby("family"):
        family_data = family_data.sort_values("date").set_index("date")

        for period_name, (start_date, end_date) in OOS_PERIODS.items():
            period_ic = family_data["ic"].loc[start_date:end_date]

            for regime_name, state_labels in REGIME_DEFINITIONS.items():
                result = regime_difference_test(
                    period_ic,
                    regimes[regime_name].loc[start_date:end_date],
                    maxlags=125,
                )
                rows.append(
                    {
                        "analysis": "walk_forward_oos",
                        "period": period_name,
                        "factor": family,
                        "horizon_days": "dynamic",
                        "regime": regime_name,
                        "state_0_label": state_labels[0],
                        "state_1_label": state_labels[1],
                        **result,
                    }
                )

    return pd.DataFrame(rows)


def plot_oos_regime_differences(results):
    data = results[
        (results["analysis"] == "walk_forward_oos")
        & (results["period"] == "oos_full")
    ]
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for axis, (regime_name, regime_data) in zip(
        axes,
        data.groupby("regime", sort=False),
    ):
        regime_data = regime_data.sort_values("difference_state_1_minus_0")
        axis.bar(
            regime_data["factor"],
            regime_data["difference_state_1_minus_0"],
        )
        axis.axhline(0, color="black", linewidth=0.8)
        state_0, state_1 = REGIME_DEFINITIONS[regime_name]
        axis.set_title(f"{regime_name}: Mean IC ({state_1} minus {state_0})")
        axis.set_ylabel("IC difference")
        axis.grid(axis="y", alpha=0.25)

    axes[-1].tick_params(axis="x", rotation=45)
    figure.suptitle("Purged walk-forward OOS factor sensitivity to market regimes")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "walk_forward_oos_regime_differences.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_regime_research():
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    selected, selected_factors = load_selected_factors()
    baselines = build_baseline_factors(returns, prices, availability)
    regimes = build_point_in_time_regimes(returns, prices, availability)
    fixed_results = fixed_factor_regime_tests(
        selected,
        selected_factors,
        baselines,
        prices,
        membership,
        regimes,
    )

    oos_path = os.path.join(WALK_FORWARD_OUTPUT_DIR, "stitched_oos_ic.parquet")
    if not os.path.exists(oos_path):
        raise FileNotFoundError("Run walk_forward.py before regime_research.py")
    oos_ic = pd.read_parquet(oos_path)
    oos_results = walk_forward_regime_tests(oos_ic, regimes)
    results = add_regime_multiple_testing(
        pd.concat([fixed_results, oos_results], ignore_index=True)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    regimes.to_parquet(os.path.join(OUTPUT_DIR, "point_in_time_regimes.parquet"))
    results.to_csv(os.path.join(OUTPUT_DIR, "regime_statistics.csv"), index=False)
    plot_oos_regime_differences(results)
    return regimes, results


def print_regime_results(results):
    oos = results[
        (results["analysis"] == "walk_forward_oos")
        & (results["period"] == "oos_full")
    ]
    columns = [
        "factor",
        "regime",
        "state_0_mean_ic",
        "state_1_mean_ic",
        "difference_state_1_minus_0",
        "difference_hac_tstat",
        "difference_pvalue_fdr_bh",
    ]
    print("\nSTAGE 6: WALK-FORWARD OOS REGIME DIFFERENCES")
    print(
        oos[columns].sort_values(
            ["regime", "difference_state_1_minus_0"],
            ascending=[True, False],
        ).to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    _, regime_results = run_regime_research()
    print_regime_results(regime_results)
