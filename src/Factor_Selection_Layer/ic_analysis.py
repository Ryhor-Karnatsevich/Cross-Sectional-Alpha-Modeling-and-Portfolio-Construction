import numpy as np
import pandas as pd

from selection_config import (
    FORWARD_HORIZONS,
    MIN_ASSETS,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    SIGNAL_LAG,
)
from selection_storage import (
    load_factor_matrix,
    load_factor_metadata,
    load_forward_return_matrices,
    load_membership,
)


# -------------------------
# DAILY IC
def compute_daily_ic(factor, forward_returns, membership):
    lagged_factor = factor.shift(SIGNAL_LAG).where(membership)
    valid_assets = lagged_factor.notna() & forward_returns.notna()
    factor_ranks = lagged_factor.where(valid_assets).rank(
        axis=1,
        method="average",
    )
    return_ranks = forward_returns.where(valid_assets).rank(
        axis=1,
        method="average",
    )
    ic = factor_ranks.corrwith(return_ranks, axis=1)
    asset_count = valid_assets.sum(axis=1)
    return ic.where(asset_count >= MIN_ASSETS)


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
# OPTIONAL IC DATASET
def hypothesis_key(family, variant, horizon):
    return f"{family}|{variant}|h{int(horizon)}"


def build_daily_ic():
    factor_metadata = load_factor_metadata()
    membership = load_membership()
    forward_returns = load_forward_return_matrices(
        FORWARD_HORIZONS,
        membership,
    )
    daily_ic_columns = {}
    hypothesis_rows = []

    for factor_information in factor_metadata.itertuples(index=False):
        factor = load_factor_matrix(
            factor_information.family,
            factor_information.variant,
            membership,
        )
        print(
            "Optional IC: "
            f"{factor_information.family} | {factor_information.variant}"
        )

        for horizon in FORWARD_HORIZONS:
            key = hypothesis_key(
                factor_information.family,
                factor_information.variant,
                horizon,
            )
            daily_ic_columns[key] = compute_daily_ic(
                factor,
                forward_returns[horizon],
                membership,
            ).astype("float32")
            hypothesis_rows.append(
                {
                    "key": key,
                    "family": factor_information.family,
                    "variant": factor_information.variant,
                    "horizon_days": horizon,
                    "parameters": factor_information.parameters,
                }
            )

    return pd.DataFrame(daily_ic_columns), pd.DataFrame(hypothesis_rows)


def summarize_daily_ic(daily_ic, metadata):
    rows = []
    research_end = RESEARCH_END_DATE or daily_ic.index.max()

    for hypothesis in metadata.itertuples(index=False):
        values = daily_ic.loc[
            RESEARCH_START_DATE:research_end,
            hypothesis.key,
        ]
        rows.append(
            {
                "family": hypothesis.family,
                "variant": hypothesis.variant,
                "horizon_days": int(hypothesis.horizon_days),
                "parameters": hypothesis.parameters,
                **summarize_ic(values, int(hypothesis.horizon_days)),
            }
        )

    return pd.DataFrame(rows)
