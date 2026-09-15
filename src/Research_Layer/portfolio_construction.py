import numpy as np
import pandas as pd

from research_config import (
    GROSS_EXPOSURE,
    LONG_ONLY_EXPOSURE,
    MIDDLE_LOWER,
    MIDDLE_UPPER,
    MIN_ASSETS,
    MIN_NAMES_PER_LEG,
    TAIL_FRACTION,
)


def percentile_ranks(scores, eligible):
    valid = scores.where(eligible).dropna()
    if len(valid) < MIN_ASSETS:
        return pd.Series(dtype=float)
    return valid.rank(method="average", pct=True)


def equal_leg_weights(long_names, short_names):
    long_names = pd.Index(long_names)
    short_names = pd.Index(short_names)
    if (
        len(long_names) < MIN_NAMES_PER_LEG
        or len(short_names) < MIN_NAMES_PER_LEG
    ):
        return pd.Series(dtype=float)

    weights = pd.Series(
        0.0,
        index=long_names.union(short_names),
        dtype=float,
    )
    weights.loc[long_names] = GROSS_EXPOSURE / 2 / len(long_names)
    weights.loc[short_names] = -GROSS_EXPOSURE / 2 / len(short_names)
    return weights


def continuous_weights(ranks):
    raw = ranks - 0.5
    long = raw.clip(lower=0)
    short = (-raw.clip(upper=0))
    if (
        (long > 0).sum() < MIN_NAMES_PER_LEG
        or (short > 0).sum() < MIN_NAMES_PER_LEG
        or long.sum() <= 0
        or short.sum() <= 0
    ):
        return pd.Series(dtype=float)
    return (
        long / long.sum() * GROSS_EXPOSURE / 2
        - short / short.sum() * GROSS_EXPOSURE / 2
    )


def long_only_weights(names):
    names = pd.Index(names)
    if len(names) < MIN_NAMES_PER_LEG:
        return pd.Series(dtype=float)
    return pd.Series(LONG_ONLY_EXPOSURE / len(names), index=names)


def beta_neutralize(weights, betas):
    if weights.empty:
        return weights, False
    aligned_betas = betas.reindex(weights.index)
    weights = weights.loc[aligned_betas.notna()]
    aligned_betas = aligned_betas.dropna()
    long = weights[weights > 0]
    short = -weights[weights < 0]
    if (
        len(long) < MIN_NAMES_PER_LEG
        or len(short) < MIN_NAMES_PER_LEG
    ):
        return weights, False

    long_betas = aligned_betas.reindex(long.index)
    short_betas = aligned_betas.reindex(short.index)

    long_average_beta = float((long / long.sum() * long_betas).sum())
    short_average_beta = float((short / short.sum() * short_betas).sum())
    denominator = long_average_beta + short_average_beta
    if (
        not np.isfinite(denominator)
        or denominator <= 0
        or long_average_beta <= 0
        or short_average_beta <= 0
    ):
        return weights, False

    long_exposure = GROSS_EXPOSURE * short_average_beta / denominator
    short_exposure = GROSS_EXPOSURE * long_average_beta / denominator
    adjusted = pd.Series(0.0, index=weights.index)
    adjusted.loc[long.index] = long / long.sum() * long_exposure
    adjusted.loc[short.index] = -short / short.sum() * short_exposure
    return adjusted, True


def build_target_weights_from_ranks(ranks, method, betas, beta_neutral):
    if ranks.empty:
        return pd.Series(dtype=float), False

    q1 = ranks <= TAIL_FRACTION
    q10 = ranks > 1 - TAIL_FRACTION
    middle = (ranks > MIDDLE_LOWER) & (ranks <= MIDDLE_UPPER)

    if method == "continuous_high_minus_low":
        weights = continuous_weights(ranks)
    elif method == "q10_minus_q1":
        weights = equal_leg_weights(ranks.index[q10], ranks.index[q1])
    elif method == "q10_minus_middle":
        weights = equal_leg_weights(ranks.index[q10], ranks.index[middle])
    elif method == "middle_minus_q1":
        weights = equal_leg_weights(ranks.index[middle], ranks.index[q1])
    elif method == "long_q10":
        weights = long_only_weights(ranks.index[q10])
    elif method == "long_q1":
        weights = long_only_weights(ranks.index[q1])
    else:
        raise ValueError(f"Unknown portfolio method: {method}")

    if beta_neutral:
        adjusted, applied = beta_neutralize(weights, betas)
        if not applied:
            return pd.Series(dtype=float), False
        return adjusted, True
    return weights, False


def build_target_weights(scores, eligible, method, betas, beta_neutral):
    ranks = percentile_ranks(scores, eligible)
    return build_target_weights_from_ranks(
        ranks,
        method,
        betas,
        beta_neutral,
    )
