import json

import numpy as np
import pandas as pd

from factor_config import (
    ANNUALIZATION_FACTOR,
    APPLY_WINSORIZATION,
    FACTOR_CONFIGS,
    FORWARD_HORIZONS,
    MIN_ASSETS,
    MIN_OBSERVATION_RATIO,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    SIGNAL_LAG,
    WINSOR_LOWER,
    WINSOR_UPPER,
)
from factor_storage import save_factor_matrix, save_sensitivity_cache
from factors import (
    compute_high_proximity,
    compute_liquidity_change,
    compute_low_volatility,
    compute_momentum,
    compute_price_volume_confirmation,
    compute_residual_momentum,
    compute_risk_adjusted_trend,
    compute_short_term_reversal,
    compute_trend,
    compute_trend_slope,
    compute_volatility_scaled_momentum,
)
from transforms import prepare_factor


# -------------------------
# SETTINGS PREPARATION
def required_observations(window):
    return max(2, int(np.ceil(window * MIN_OBSERVATION_RATIO)))


def configuration_parameters(configuration):
    return {
        key: value
        for key, value in configuration.items()
        if key != "variant"
    }


# -------------------------
# FACTOR BUILDING
def prepare_raw_factor(raw_factor, availability):
    return prepare_factor(
        raw_factor,
        availability,
        APPLY_WINSORIZATION,
        WINSOR_LOWER,
        WINSOR_UPPER,
    )


def build_factor_scores(family, configuration, inputs):
    returns = inputs["returns"]
    prices = inputs["prices"]
    volume = inputs["volume"]
    availability = inputs["availability"]
    window = configuration.get("window")
    skip = configuration.get("skip", 0)

    if family == "momentum":
        raw = compute_momentum(
            returns,
            window,
            skip,
            required_observations(window - skip),
        )

    elif family == "low_volatility":
        raw = compute_low_volatility(
            returns,
            window,
            required_observations(window),
        )

    elif family == "trend":
        raw = compute_trend(
            prices,
            window,
            required_observations(window),
        )

    elif family == "short_term_reversal":
        raw = compute_short_term_reversal(
            returns,
            window,
            required_observations(window),
        )

    elif family == "residual_momentum":
        raw = compute_residual_momentum(
            returns,
            availability,
            window,
            skip,
            required_observations(window - skip),
        )

    elif family == "volatility_scaled_momentum":
        volatility_window = configuration["volatility_window"]
        raw = compute_volatility_scaled_momentum(
            returns,
            window,
            skip,
            required_observations(window - skip),
            volatility_window,
            required_observations(volatility_window),
        )

    elif family == "high_proximity":
        raw = compute_high_proximity(
            prices,
            window,
            required_observations(window),
        )

    elif family == "trend_slope":
        raw = compute_trend_slope(
            prices,
            window,
            required_observations(window),
            ANNUALIZATION_FACTOR,
        )

    elif family == "risk_adjusted_trend":
        raw = compute_risk_adjusted_trend(
            prices,
            returns,
            window,
            required_observations(window),
            ANNUALIZATION_FACTOR,
        )

    elif family == "liquidity_change":
        short_window = configuration["short_window"]
        long_window = configuration["long_window"]
        raw = compute_liquidity_change(
            prices,
            volume,
            short_window,
            long_window,
            required_observations(short_window),
            required_observations(long_window),
        )

    elif family == "price_volume_confirmation":
        short_window = configuration["short_window"]
        long_window = configuration["long_window"]
        momentum = prepare_raw_factor(
            compute_momentum(
                returns,
                window,
                skip,
                required_observations(window - skip),
            ),
            availability,
        )
        liquidity = prepare_raw_factor(
            compute_liquidity_change(
                prices,
                volume,
                short_window,
                long_window,
                required_observations(short_window),
                required_observations(long_window),
            ),
            availability,
        )
        raw = compute_price_volume_confirmation(
            momentum,
            liquidity,
            configuration["confirmation_strength"],
            configuration["liquidity_clip"],
        )

    else:
        raise ValueError(f"Unknown factor family: {family}")

    return prepare_raw_factor(raw, availability)


# -------------------------
# FORWARD RETURNS
def compute_forward_returns(prices, price_quality, horizon):
    future_prices = prices.shift(-horizon)
    future_quality = price_quality.shift(-horizon, fill_value=False)
    valid = (
        price_quality
        & future_quality
        & prices.notna()
        & future_prices.notna()
    )
    return (future_prices / prices - 1).where(valid)


# -------------------------
# DAILY IC
def compute_daily_ic(factor, forward_returns, membership):
    lagged_factor = factor.shift(SIGNAL_LAG)
    lagged_factor, forward_returns = lagged_factor.align(
        forward_returns,
        join="inner",
    )
    membership = membership.reindex(
        index=lagged_factor.index,
        columns=lagged_factor.columns,
        fill_value=False,
    )
    lagged_factor = lagged_factor.where(membership)
    forward_returns = forward_returns.where(membership)
    valid_assets = lagged_factor.notna() & forward_returns.notna()
    lagged_factor = lagged_factor.where(valid_assets)
    forward_returns = forward_returns.where(valid_assets)
    factor_ranks = lagged_factor.rank(axis=1, method="average")
    return_ranks = forward_returns.rank(axis=1, method="average")
    ic = factor_ranks.corrwith(return_ranks, axis=1)
    asset_count = valid_assets.sum(axis=1)
    ic = ic.where(asset_count >= MIN_ASSETS)

    return pd.DataFrame({"ic": ic, "asset_count": asset_count})


# -------------------------
# IC STATISTICS
def hac_tstat(values, max_lag):
    values = pd.Series(values).dropna().astype(float)
    observation_count = len(values)

    if observation_count < 2:
        return np.nan

    demeaned = values.to_numpy() - values.mean()
    max_lag = min(max_lag, observation_count - 1)
    long_run_variance = np.dot(demeaned, demeaned) / observation_count

    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        covariance = np.dot(demeaned[lag:], demeaned[:-lag]) / observation_count
        long_run_variance += 2 * weight * covariance

    if long_run_variance <= 0:
        return np.nan

    standard_error = np.sqrt(long_run_variance / observation_count)
    return values.mean() / standard_error


def summarize_ic(values, horizon):
    values = pd.Series(values).dropna()

    return {
        "observations": len(values),
        "mean_ic": values.mean() if not values.empty else np.nan,
        "std_ic": values.std() if len(values) > 1 else np.nan,
        "tstat": hac_tstat(values, horizon - 1),
        "positive_rate": (
            (values > 0).mean()
            if not values.empty
            else np.nan
        ),
    }


# -------------------------
# SENSITIVITY
def hypothesis_key(family, variant, horizon):
    return f"{family}|{variant}|h{horizon}"


def run_sensitivity(inputs):
    prices = inputs["prices"]
    price_quality = inputs["price_quality"]
    membership = inputs["membership"]
    research_end = RESEARCH_END_DATE or prices.index.max()
    research_slice = slice(RESEARCH_START_DATE, research_end)
    forward_returns = {
        horizon: compute_forward_returns(prices, price_quality, horizon)
        for horizon in FORWARD_HORIZONS
    }
    daily_ic_columns = {}
    metadata_rows = []
    result_rows = []

    for family, configurations in FACTOR_CONFIGS.items():
        for configuration in configurations:
            variant = configuration["variant"]
            print(f"Sensitivity: {family} | {variant}")
            factor = build_factor_scores(family, configuration, inputs)
            save_factor_matrix(family, variant, factor)

            for horizon in FORWARD_HORIZONS:
                key = hypothesis_key(family, variant, horizon)
                ic_data = compute_daily_ic(
                    factor,
                    forward_returns[horizon],
                    membership,
                )
                daily_ic_columns[key] = ic_data["ic"].astype("float32")
                parameters = json.dumps(
                    configuration_parameters(configuration),
                    sort_keys=True,
                )
                metadata_rows.append(
                    {
                        "key": key,
                        "family": family,
                        "variant": variant,
                        "horizon_days": horizon,
                        "parameters": parameters,
                    }
                )
                statistics = summarize_ic(
                    ic_data.loc[research_slice, "ic"],
                    horizon,
                )
                result_rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "horizon_days": horizon,
                        "parameters": parameters,
                        **statistics,
                    }
                )

    daily_ic = pd.DataFrame(daily_ic_columns)
    metadata = pd.DataFrame(metadata_rows)
    sensitivity_results = pd.DataFrame(result_rows)
    save_sensitivity_cache(daily_ic, metadata)

    return sensitivity_results, daily_ic, metadata
