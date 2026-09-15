from itertools import combinations

import numpy as np
import pandas as pd

from research_config import MIN_ASSETS, RESEARCH_START_DATE, SIGNAL_LAG


def rowwise_correlation(left, right):
    left = left.to_numpy(dtype=float, na_value=np.nan)
    right = right.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(left) & np.isfinite(right)
    counts = valid.sum(axis=1)
    left = np.where(valid, left, 0.0)
    right = np.where(valid, right, 0.0)
    safe_count = np.where(counts > 0, counts, 1)
    left_mean = left.sum(axis=1) / safe_count
    right_mean = right.sum(axis=1) / safe_count
    left_centered = np.where(valid, left - left_mean[:, None], 0.0)
    right_centered = np.where(valid, right - right_mean[:, None], 0.0)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt(
        (left_centered**2).sum(axis=1)
        * (right_centered**2).sum(axis=1)
    )
    result = np.full(len(counts), np.nan)
    usable = (counts >= MIN_ASSETS) & (denominator > 0)
    result[usable] = numerator[usable] / denominator[usable]
    return result, counts


def run_factor_overlap(factors, membership, availability):
    usable = membership & availability
    ranks = {
        key: factor.shift(SIGNAL_LAG)
        .where(usable)
        .rank(axis=1, pct=True)
        .loc[RESEARCH_START_DATE:]
        for key, factor in factors.items()
    }
    rows = []
    for left_key, right_key in combinations(ranks, 2):
        correlation, counts = rowwise_correlation(
            ranks[left_key],
            ranks[right_key],
        )
        valid = pd.Series(correlation).dropna()
        rows.append(
            {
                "factor_left": left_key,
                "factor_right": right_key,
                "observations": len(valid),
                "median_common_assets": float(
                    np.median(counts[counts >= MIN_ASSETS])
                ),
                "mean_daily_rank_correlation": valid.mean(),
                "median_daily_rank_correlation": valid.median(),
                "mean_absolute_daily_rank_correlation": valid.abs().mean(),
                "high_absolute_overlap_rate": (valid.abs() >= 0.80).mean(),
            }
        )
    return pd.DataFrame(rows)
