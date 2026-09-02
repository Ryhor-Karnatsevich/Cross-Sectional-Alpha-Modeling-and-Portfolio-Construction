"""Sensitivity research for cross-sectional factors.

The same predefined factor variants are evaluated on train, validation, test,
and rolling 36-observation IC windows. Test metrics are reported but are never
used to select parameters.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors import compute_momentum, compute_trend, compute_volatility
from pipeline import build_factor, compute_ic, load_data


# -----------------------------------------------------------------------------
# FROZEN RESEARCH RULES
# -----------------------------------------------------------------------------
REBALANCE_STEP = 21
SIGNAL_LAG = 1
MIN_OBSERVATION_RATIO = 0.80
ROLLING_IC_WINDOW = 36
ROLLING_IC_MIN_PERIODS = 24

PERIODS = {
    "train": ("2010-01-01", "2017-12-31"),
    "validation": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-12-31"),
}

MOMENTUM_WINDOWS = [63, 126, 189, 252, 378, 504]
MOMENTUM_SKIPS = [0, 21, 42]
LOW_VOL_WINDOWS = [20, 40, 60, 90, 120, 180, 252]
TREND_WINDOWS = [20, 50, 100, 150, 200, 250]


# Selection uses train and validation only. Test is excluded by construction.
SELECTION_WEIGHTS = {
    "validation_mean_ic": 0.25,
    "validation_tstat": 0.20,
    "validation_positive_rate": 0.15,
    "stability": 0.15,
    "rolling_positive_rate": 0.15,
    "train_mean_ic": 0.10,
}


def required_observations(window):
    """Use one data-quality rule for every factor specification."""
    return max(2, int(np.ceil(window * MIN_OBSERVATION_RATIO)))


def create_parameter_grid():
    """Create the frozen, economically interpretable sensitivity grid."""
    momentum = []

    for window in MOMENTUM_WINDOWS:
        for skip in MOMENTUM_SKIPS:
            formation_window = window - skip

            # Keep at least two months of actual formation returns.
            if formation_window < 42:
                continue

            momentum.append(
                {
                    "name": f"{window // 21}m-{skip // 21}m",
                    "window": window,
                    "skip": skip,
                    "min_obs": required_observations(formation_window),
                }
            )

    low_vol = [
        {
            "name": f"{window}d",
            "window": window,
            "min_obs": required_observations(window),
        }
        for window in LOW_VOL_WINDOWS
    ]

    trend = [
        {
            "name": f"SMA{window}",
            "window": window,
            "min_obs": required_observations(window),
        }
        for window in TREND_WINDOWS
    ]

    return {
        "momentum": momentum,
        "low_vol": low_vol,
        "trend": trend,
    }


PARAMETER_GRID = create_parameter_grid()


# -----------------------------------------------------------------------------
# FACTOR BUILDING
# -----------------------------------------------------------------------------
def factor_parameters(parameters):
    """Remove the display name before passing parameters into a formula."""
    return {key: value for key, value in parameters.items() if key != "name"}


def build_factor_variant(factor_name, parameters, returns, prices, availability):
    """Build one normalized factor variant using the existing formulas."""
    parameters = factor_parameters(parameters)

    if factor_name == "momentum":
        raw = compute_momentum(returns, **parameters)
        return build_factor(raw, availability)

    if factor_name == "low_vol":
        raw = compute_volatility(returns, **parameters)
        return -build_factor(raw, availability)

    if factor_name == "trend":
        raw = compute_trend(prices, **parameters)
        return build_factor(raw, availability)

    raise ValueError(f"Unknown factor: {factor_name}")


def find_variant_parameters(factor_name, variant_name):
    """Return the frozen parameters for one named variant."""
    for parameters in PARAMETER_GRID[factor_name]:
        if parameters["name"] == variant_name:
            return parameters

    raise ValueError(f"Unknown variant: {factor_name} | {variant_name}")


# -----------------------------------------------------------------------------
# IC METRICS
# -----------------------------------------------------------------------------
def calculate_metrics(ic, start_date, end_date):
    """Calculate IC statistics for one inclusive time period."""
    period_ic = ic.loc[start_date:end_date].dropna()

    if period_ic.empty:
        return {
            "observations": 0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "tstat": np.nan,
            "positive_rate": np.nan,
        }

    mean_ic = period_ic.mean()
    std_ic = period_ic.std()
    tstat = mean_ic / std_ic * np.sqrt(len(period_ic)) if std_ic != 0 else np.nan

    return {
        "observations": len(period_ic),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "tstat": tstat,
        "positive_rate": (period_ic > 0).mean(),
    }


def calculate_rolling_ic(ic, end_date):
    """Calculate the full rolling IC path up to the requested boundary."""
    return (
        ic.loc[:end_date]
        .rolling(ROLLING_IC_WINDOW, min_periods=ROLLING_IC_MIN_PERIODS)
        .mean()
    )


def add_selection_score(results):
    """Score variants within a factor using train and validation only."""
    results = results.copy()
    results["stability_gap"] = (
        results["train_mean_ic"] - results["validation_mean_ic"]
    ).abs()
    results["selection_eligible"] = (
        (results["train_mean_ic"] > 0)
        & (results["validation_mean_ic"] > 0)
    )
    results["selection_score"] = np.nan

    for factor_name, factor_results in results.groupby("factor"):
        index = factor_results.index

        validation_mean_rank = factor_results["validation_mean_ic"].rank(pct=True)
        validation_tstat_rank = factor_results["validation_tstat"].rank(pct=True)
        validation_positive_rank = factor_results[
            "validation_positive_rate"
        ].rank(pct=True)
        stability_rank = (-factor_results["stability_gap"]).rank(pct=True)
        rolling_positive_rank = factor_results["rolling_positive_rate"].rank(
            pct=True
        )
        train_mean_rank = factor_results["train_mean_ic"].rank(pct=True)

        score = (
            SELECTION_WEIGHTS["validation_mean_ic"] * validation_mean_rank
            + SELECTION_WEIGHTS["validation_tstat"] * validation_tstat_rank
            + SELECTION_WEIGHTS["validation_positive_rate"]
            * validation_positive_rank
            + SELECTION_WEIGHTS["stability"] * stability_rank
            + SELECTION_WEIGHTS["rolling_positive_rate"] * rolling_positive_rank
            + SELECTION_WEIGHTS["train_mean_ic"] * train_mean_rank
        )

        results.loc[index, "selection_score"] = score

    return results


def select_best_variants(results):
    """Select one variant per factor without using any test columns."""
    selected_rows = []

    for factor_name, factor_results in results.groupby("factor"):
        ranked = factor_results.sort_values(
            ["selection_eligible", "selection_score"],
            ascending=[False, False],
        )
        selected_rows.append(ranked.iloc[0])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


# -----------------------------------------------------------------------------
# SENSITIVITY PIPELINE
# -----------------------------------------------------------------------------
def run_factor_sensitivity(returns, prices, availability, forward_returns):
    """Run the same variants through periods and rolling-window analysis."""
    rows = []
    rolling_ic = {}

    for factor_name, variants in PARAMETER_GRID.items():
        for parameters in variants:
            factor = build_factor_variant(
                factor_name,
                parameters,
                returns,
                prices,
                availability,
            )
            ic = compute_ic(
                factor,
                forward_returns,
                rebalance_step=REBALANCE_STEP,
                signal_lag=SIGNAL_LAG,
            )

            key = f"{factor_name} | {parameters['name']}"
            rolling = calculate_rolling_ic(ic, PERIODS["test"][1])
            validation_rolling = rolling.loc[: PERIODS["validation"][1]].dropna()

            row = {
                "factor": factor_name,
                "variant": parameters["name"],
                "parameters": ", ".join(
                    f"{key}={value}"
                    for key, value in parameters.items()
                    if key != "name"
                ),
                "rolling_positive_rate": (
                    (validation_rolling > 0).mean()
                    if not validation_rolling.empty
                    else np.nan
                ),
            }

            for period_name, (start_date, end_date) in PERIODS.items():
                metrics = calculate_metrics(ic, start_date, end_date)

                for metric_name, value in metrics.items():
                    row[f"{period_name}_{metric_name}"] = value

            rows.append(row)
            rolling_ic[key] = rolling

    results = add_selection_score(pd.DataFrame(rows))

    return results, pd.DataFrame(rolling_ic)


# -----------------------------------------------------------------------------
# EQUAL-WEIGHT COMBINATION CHECK
# -----------------------------------------------------------------------------
def build_selected_factors(selected, returns, prices, availability):
    """Rebuild only the selected factor specifications."""
    selected_factors = {}

    for row in selected.itertuples():
        parameters = find_variant_parameters(row.factor, row.variant)
        selected_factors[row.factor] = build_factor_variant(
            row.factor,
            parameters,
            returns,
            prices,
            availability,
        )

    return selected_factors


def equal_weight_factor_combination(selected_factors, factor_names):
    """Average normalized factors with equal weights."""
    combined = selected_factors[factor_names[0]].copy()

    for factor_name in factor_names[1:]:
        combined = combined + selected_factors[factor_name]

    return combined / len(factor_names)


def run_combination_check(
    selected,
    returns,
    prices,
    availability,
    forward_returns,
):
    """Check whether selected factors become stronger when combined."""
    selected_factors = build_selected_factors(
        selected,
        returns,
        prices,
        availability,
    )
    factor_names = list(selected_factors)
    combination_names = list(combinations(factor_names, 2))
    combination_names.append(tuple(factor_names))

    rows = []
    rolling_ic = {}

    for names in combination_names:
        combined = equal_weight_factor_combination(selected_factors, names)
        ic = compute_ic(
            combined,
            forward_returns,
            rebalance_step=REBALANCE_STEP,
            signal_lag=SIGNAL_LAG,
        )
        combination_name = " + ".join(names)

        row = {"combination": combination_name}

        for period_name, (start_date, end_date) in PERIODS.items():
            metrics = calculate_metrics(ic, start_date, end_date)

            for metric_name, value in metrics.items():
                row[f"{period_name}_{metric_name}"] = value

        rows.append(row)
        rolling_ic[combination_name] = calculate_rolling_ic(
            ic,
            PERIODS["test"][1],
        )

    return pd.DataFrame(rows), pd.DataFrame(rolling_ic)


def run_research():
    """Run the complete frozen factor research experiment."""
    returns, availability, forward_returns, prices = load_data()
    results, rolling_ic = run_factor_sensitivity(
        returns,
        prices,
        availability,
        forward_returns,
    )
    selected = select_best_variants(results)
    combination_results, combination_rolling_ic = run_combination_check(
        selected,
        returns,
        prices,
        availability,
        forward_returns,
    )

    return (
        results,
        rolling_ic,
        selected,
        combination_results,
        combination_rolling_ic,
    )


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
def print_factor_results(results):
    """Print one comparable row for every sensitivity variant."""
    columns = [
        "factor",
        "variant",
        "train_mean_ic",
        "train_tstat",
        "validation_mean_ic",
        "validation_tstat",
        "test_mean_ic",
        "test_tstat",
        "rolling_positive_rate",
        "selection_eligible",
        "selection_score",
    ]
    display = results[columns].sort_values(
        ["factor", "selection_eligible", "selection_score"],
        ascending=[True, False, False],
    )

    print("\nFACTOR SENSITIVITY RESULTS")
    print("Test is reported but is not used in selection_score.")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_selected_variants(selected):
    """Print the specifications selected from train and validation."""
    columns = [
        "factor",
        "variant",
        "parameters",
        "selection_eligible",
        "selection_score",
    ]

    print("\nSELECTED FROM TRAIN + VALIDATION")
    print(selected[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_combination_results(results):
    """Print equal-weight combined-alpha IC results."""
    columns = [
        "combination",
        "train_mean_ic",
        "train_tstat",
        "validation_mean_ic",
        "validation_tstat",
        "test_mean_ic",
        "test_tstat",
    ]

    print("\nEQUAL-WEIGHT COMBINATION RESULTS")
    print(results[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_rolling_heatmaps(rolling_ic):
    """Show every variant and every rolling-window value without line clutter."""
    factor_names = list(PARAMETER_GRID)
    figure, axes = plt.subplots(
        len(factor_names),
        1,
        figsize=(15, 12),
        sharex=True,
        layout="constrained",
    )
    values = rolling_ic.to_numpy()
    color_limit = np.nanquantile(np.abs(values), 0.98)
    image = None

    for axis, factor_name in zip(axes, factor_names):
        columns = [
            column
            for column in rolling_ic.columns
            if column.startswith(f"{factor_name} |")
        ]
        data = rolling_ic[columns].T
        image = axis.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set_yticks(range(len(columns)))
        axis.set_yticklabels(
            [column.split(" | ", maxsplit=1)[1] for column in columns]
        )
        axis.set_title(f"{factor_name}: rolling {ROLLING_IC_WINDOW}-observation Mean IC")

    tick_positions = np.linspace(0, len(rolling_ic.index) - 1, 9, dtype=int)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(
        [rolling_ic.index[position].strftime("%Y-%m") for position in tick_positions]
    )
    axes[-1].set_xlabel("Rolling IC date")
    figure.colorbar(
        image,
        ax=axes,
        label="Mean IC",
        shrink=0.85,
        pad=0.02,
        fraction=0.025,
    )
    figure.suptitle("Factor sensitivity through time")


def plot_combination_rolling_ic(rolling_ic):
    """Show the rolling IC of equal-weight factor combinations."""
    figure, axis = plt.subplots(figsize=(13, 5))

    for column in rolling_ic.columns:
        axis.plot(rolling_ic.index, rolling_ic[column], label=column)

    axis.axhline(0, color="black", linewidth=1)
    axis.set_title(f"Combined alpha: rolling {ROLLING_IC_WINDOW}-observation Mean IC")
    axis.set_xlabel("Date")
    axis.set_ylabel("Mean IC")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()


if __name__ == "__main__":
    (
        results,
        rolling_ic,
        selected,
        combination_results,
        combination_rolling_ic,
    ) = run_research()

    print_factor_results(results)
    print_selected_variants(selected)
    print_combination_results(combination_results)
    plot_rolling_heatmaps(rolling_ic)
    plot_combination_rolling_ic(combination_rolling_ic)
    plt.show()
