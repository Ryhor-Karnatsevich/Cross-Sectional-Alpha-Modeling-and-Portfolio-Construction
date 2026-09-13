import json

import numpy as np
import pandas as pd

from factor_config import (
    ANNUALIZATION_FACTOR,
    APPLY_WINSORIZATION,
    FACTOR_CONFIGS,
    MIN_OBSERVATION_RATIO,
    WINSOR_LOWER,
    WINSOR_UPPER,
)
from factor_storage import save_factor_matrix
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


def factor_key(family, variant):
    return f"{family}|{variant}"


# -------------------------
# FACTOR PREPARATION
def prepare_raw_factor(raw_factor, availability):
    return prepare_factor(
        raw_factor,
        availability,
        APPLY_WINSORIZATION,
        WINSOR_LOWER,
        WINSOR_UPPER,
    )


# -------------------------
# FACTOR BUILDING
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
# COMPLETE FACTOR MATRIX SET
def build_factor_matrices(inputs):
    metadata_rows = []

    for family, configurations in FACTOR_CONFIGS.items():
        for configuration in configurations:
            variant = configuration["variant"]
            print(f"Factor matrix: {family} | {variant}")
            factor = build_factor_scores(family, configuration, inputs)
            path = save_factor_matrix(family, variant, factor)
            metadata_rows.append(
                {
                    "key": factor_key(family, variant),
                    "family": family,
                    "variant": variant,
                    "parameters": json.dumps(
                        configuration_parameters(configuration),
                        sort_keys=True,
                    ),
                    "path": path,
                }
            )

    metadata = pd.DataFrame(metadata_rows)

    if len(metadata) != 56 or metadata["key"].duplicated().any():
        raise ValueError("Factor metadata must contain 56 unique matrices")

    return metadata
