import numpy as np
import pandas as pd

from research_config import (
    BETA_LOOKBACK,
    BETA_MIN_OBSERVATIONS,
    REGIME_CORRELATION_WINDOW,
    REGIME_DISPERSION_WINDOW,
    REGIME_EXPANDING_MIN_OBSERVATIONS,
    REGIME_VOLATILITY_WINDOW,
    HIGH_RISK_FREE_RATE_PCT,
)


def build_market_return(returns, membership, availability):
    usable = returns.where(membership & availability)
    return usable.mean(axis=1)


def build_trailing_betas(returns, market_return):
    covariance = returns.rolling(
        BETA_LOOKBACK,
        min_periods=BETA_MIN_OBSERVATIONS,
    ).cov(market_return)
    market_variance = market_return.rolling(
        BETA_LOOKBACK,
        min_periods=BETA_MIN_OBSERVATIONS,
    ).var()
    return covariance.div(market_variance, axis=0).shift(1)


def expanding_binary_state(series, minimum_observations, high_name):
    threshold = (
        series.expanding(min_periods=minimum_observations)
        .median()
        .shift(1)
    )
    state = pd.Series("unavailable", index=series.index, dtype="object")
    valid = series.notna() & threshold.notna()
    state.loc[valid & series.gt(threshold)] = f"high_{high_name}"
    state.loc[valid & series.le(threshold)] = f"low_{high_name}"
    return state, threshold


def build_market_regimes(
    returns,
    membership,
    availability,
    market_return,
    risk_free,
):
    usable = returns.where(membership & availability)
    known_risk_free_rate = risk_free["annual_rate_pct"].shift(1)
    market_volatility = (
        market_return.rolling(REGIME_VOLATILITY_WINDOW).std()
        * np.sqrt(252)
    ).shift(1)
    dispersion = (
        usable.std(axis=1)
        .rolling(REGIME_DISPERSION_WINDOW)
        .mean()
        .shift(1)
    )
    average_stock_variance = usable.rolling(
        REGIME_CORRELATION_WINDOW
    ).var().mean(axis=1)
    market_variance = market_return.rolling(REGIME_CORRELATION_WINDOW).var()
    member_count = usable.notna().sum(axis=1).rolling(
        REGIME_CORRELATION_WINDOW
    ).median()
    correlation_proxy = (
        (
            member_count * market_variance.div(average_stock_variance) - 1
        ).div(member_count - 1)
        .replace([np.inf, -np.inf], np.nan)
        .clip(-1, 1)
        .shift(1)
    )

    volatility_state, volatility_threshold = expanding_binary_state(
        market_volatility,
        REGIME_EXPANDING_MIN_OBSERVATIONS,
        "volatility",
    )
    dispersion_state, dispersion_threshold = expanding_binary_state(
        dispersion,
        REGIME_EXPANDING_MIN_OBSERVATIONS,
        "dispersion",
    )
    correlation_state, correlation_threshold = expanding_binary_state(
        correlation_proxy,
        REGIME_EXPANDING_MIN_OBSERVATIONS,
        "correlation",
    )
    risk_free_state = pd.Series(
        "unavailable",
        index=returns.index,
        dtype="object",
    )
    known_rate = known_risk_free_rate.notna()
    risk_free_state.loc[
        known_rate & known_risk_free_rate.gt(HIGH_RISK_FREE_RATE_PCT)
    ] = "above_2pct"
    risk_free_state.loc[
        known_rate & known_risk_free_rate.le(HIGH_RISK_FREE_RATE_PCT)
    ] = "at_or_below_2pct"

    return pd.DataFrame(
        {
            "risk_free_rate_pct": known_risk_free_rate,
            "risk_free_state": risk_free_state,
            "market_volatility": market_volatility,
            "market_volatility_threshold": volatility_threshold,
            "volatility_state": volatility_state,
            "cross_sectional_dispersion": dispersion,
            "dispersion_threshold": dispersion_threshold,
            "dispersion_state": dispersion_state,
            "average_correlation_proxy": correlation_proxy,
            "correlation_threshold": correlation_threshold,
            "correlation_state": correlation_state,
        },
        index=returns.index,
    )
