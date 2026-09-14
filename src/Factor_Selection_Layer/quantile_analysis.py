from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import ndimage

from selection_config import (
    DAILY_QUANTILE_RESULTS_PATH,
    FORWARD_HORIZONS,
    MIN_ASSETS,
    QUANTILE_COUNT,
    QUANTILE_RUN_METADATA_PATH,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    SIGNAL_LAG,
)
from selection_storage import (
    load_factor_matrix,
    load_factor_metadata,
    load_forward_return_matrices,
    load_membership,
    prepare_selection_directories,
    save_json,
    save_parquet_chunks,
)


# -------------------------
# SIGNAL PREPARATION
def prepare_factor_quantiles(factor, membership):
    signal = factor.shift(SIGNAL_LAG).where(membership)
    ranks = signal.rank(axis=1, method="average", pct=True)
    rank_values = ranks.to_numpy(dtype=float, na_value=np.nan)
    quantiles = np.zeros(rank_values.shape, dtype=np.int8)
    valid = np.isfinite(rank_values)
    quantiles[valid] = np.ceil(
        rank_values[valid] * QUANTILE_COUNT
    ).astype(np.int8)
    np.clip(quantiles, 0, QUANTILE_COUNT, out=quantiles)
    return signal, quantiles


# -------------------------
# INTERNAL HELPERS
def grouped_median(values, groups, counts, group_count):
    if not len(values):
        return np.full(group_count, np.nan, dtype=np.float64)

    medians = np.asarray(
        ndimage.median(
            values,
            labels=groups + 1,
            index=np.arange(1, group_count + 1),
        ),
        dtype=np.float64,
    )
    medians[counts == 0] = np.nan
    return medians


def rowwise_linear_relationship(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.sum(axis=1)
    valid_x = np.where(valid, x, 0.0)
    valid_y = np.where(valid, y, 0.0)

    sum_x = valid_x.sum(axis=1, dtype=np.float64)
    sum_y = valid_y.sum(axis=1, dtype=np.float64)
    sum_xx = np.square(valid_x).sum(axis=1, dtype=np.float64)
    sum_yy = np.square(valid_y).sum(axis=1, dtype=np.float64)
    sum_xy = np.multiply(valid_x, valid_y).sum(axis=1, dtype=np.float64)

    safe_counts = np.where(counts > 0, counts, 1)
    covariance_numerator = sum_xy - sum_x * sum_y / safe_counts
    factor_variance_numerator = sum_xx - np.square(sum_x) / safe_counts
    return_variance_numerator = sum_yy - np.square(sum_y) / safe_counts

    correlation_denominator = np.sqrt(
        np.maximum(factor_variance_numerator, 0.0)
        * np.maximum(return_variance_numerator, 0.0)
    )
    correlation = np.full(len(x), np.nan, dtype=np.float64)
    beta = np.full(len(x), np.nan, dtype=np.float64)

    valid_correlation = (
        (counts >= MIN_ASSETS)
        & (correlation_denominator > 0)
    )
    valid_beta = (
        (counts >= MIN_ASSETS)
        & (factor_variance_numerator > 0)
    )
    correlation[valid_correlation] = (
        covariance_numerator[valid_correlation]
        / correlation_denominator[valid_correlation]
    )
    beta[valid_beta] = (
        covariance_numerator[valid_beta]
        / factor_variance_numerator[valid_beta]
    )
    return correlation, beta


# -------------------------
# DAILY FACTOR-RETURN RELATIONSHIP
def compute_daily_relationship_metrics(signal, forward_returns):
    signal_values = signal.to_numpy(dtype=np.float64, na_value=np.nan)
    return_values = forward_returns.to_numpy(dtype=np.float64, na_value=np.nan)
    pearson_correlation, factor_beta = rowwise_linear_relationship(
        signal_values,
        return_values,
    )

    valid_pairs = signal.notna() & forward_returns.notna()
    factor_ranks = signal.where(valid_pairs).rank(
        axis=1,
        method="average",
    )
    return_ranks = forward_returns.where(valid_pairs).rank(
        axis=1,
        method="average",
    )
    spearman_ic, _ = rowwise_linear_relationship(
        factor_ranks.to_numpy(dtype=np.float64, na_value=np.nan),
        return_ranks.to_numpy(dtype=np.float64, na_value=np.nan),
    )

    return {
        "spearman_ic": spearman_ic,
        "pearson_correlation": pearson_correlation,
        "factor_beta": factor_beta,
    }


def aggregate_quantile_relationships(
    signal,
    quantiles,
    forward_returns,
    date_groups=None,
):
    signal_values = signal.to_numpy(dtype=np.float64, na_value=np.nan)
    return_values = forward_returns.to_numpy(dtype=np.float64, na_value=np.nan)
    date_count, asset_count = quantiles.shape

    if (
        signal_values.shape != quantiles.shape
        or return_values.shape != quantiles.shape
    ):
        raise ValueError(
            "Factor signal, quantiles and forward returns must align"
        )

    flat_quantiles = quantiles.ravel()
    flat_signal = signal_values.ravel()
    flat_returns = return_values.ravel()
    valid = (
        (flat_quantiles > 0)
        & np.isfinite(flat_signal)
        & np.isfinite(flat_returns)
    )
    if date_groups is None:
        date_groups = np.repeat(
            np.arange(date_count, dtype=np.int32) * QUANTILE_COUNT,
            asset_count,
        )
    groups = date_groups[valid] + flat_quantiles[valid].astype(np.int32) - 1
    group_count = date_count * QUANTILE_COUNT
    counts = np.bincount(groups, minlength=group_count).reshape(
        date_count,
        QUANTILE_COUNT,
    )
    signal_sums = np.bincount(
        groups,
        weights=flat_signal[valid],
        minlength=group_count,
    )
    return_sums = np.bincount(
        groups,
        weights=flat_returns[valid],
        minlength=group_count,
    )
    flat_counts = counts.ravel()
    signal_means = np.full(group_count, np.nan, dtype=np.float64)
    return_means = np.full(group_count, np.nan, dtype=np.float64)
    np.divide(
        signal_sums,
        flat_counts,
        out=signal_means,
        where=flat_counts > 0,
    )
    np.divide(
        return_sums,
        flat_counts,
        out=return_means,
        where=flat_counts > 0,
    )
    signal_medians = grouped_median(
        flat_signal[valid],
        groups,
        flat_counts,
        group_count,
    )
    return_medians = grouped_median(
        flat_returns[valid],
        groups,
        flat_counts,
        group_count,
    )

    signal_means = signal_means.reshape(date_count, QUANTILE_COUNT)
    signal_medians = signal_medians.reshape(date_count, QUANTILE_COUNT)
    return_means = return_means.reshape(date_count, QUANTILE_COUNT)
    return_medians = return_medians.reshape(date_count, QUANTILE_COUNT)

    signal_asset_count = (quantiles > 0).sum(axis=1)
    return_asset_count = counts.sum(axis=1)
    enough_assets = return_asset_count >= MIN_ASSETS
    signal_means[~enough_assets] = np.nan
    signal_medians[~enough_assets] = np.nan
    return_means[~enough_assets] = np.nan
    return_medians[~enough_assets] = np.nan

    return {
        "counts": counts,
        "signal_means": signal_means,
        "signal_medians": signal_medians,
        "return_means": return_means,
        "return_medians": return_medians,
        "signal_asset_count": signal_asset_count,
        "return_asset_count": return_asset_count,
    }


def research_date_mask(index):
    eligible = index >= pd.Timestamp(RESEARCH_START_DATE)

    if RESEARCH_END_DATE is not None:
        eligible &= index <= pd.Timestamp(RESEARCH_END_DATE)

    return eligible


def build_quantile_result_chunk(
    factor_information,
    horizon,
    index,
    membership_count,
    date_groups,
    signal,
    quantiles,
    forward_returns,
):
    quantile_statistics = aggregate_quantile_relationships(
        signal,
        quantiles,
        forward_returns,
        date_groups,
    )
    relationship_metrics = compute_daily_relationship_metrics(
        signal,
        forward_returns,
    )
    factor_key = factor_information.key
    hypothesis_key = f"{factor_key}|h{int(horizon)}"
    result = pd.DataFrame(
        {
            "date": index,
            "hypothesis_key": hypothesis_key,
            "factor_key": factor_key,
            "family": factor_information.family,
            "variant": factor_information.variant,
            "horizon_days": np.int16(horizon),
            "parameters": factor_information.parameters,
            "research_eligible": research_date_mask(index),
            "membership_count": membership_count,
            "signal_asset_count": quantile_statistics[
                "signal_asset_count"
            ].astype(np.int16),
            "return_asset_count": quantile_statistics[
                "return_asset_count"
            ].astype(np.int16),
            "spearman_ic": relationship_metrics["spearman_ic"].astype(
                np.float32
            ),
            "pearson_correlation": relationship_metrics[
                "pearson_correlation"
            ].astype(np.float32),
            "factor_beta": relationship_metrics["factor_beta"].astype(
                np.float32
            ),
        }
    )

    for quantile in range(1, QUANTILE_COUNT + 1):
        position = quantile - 1
        result[f"q{quantile}_count"] = quantile_statistics[
            "counts"
        ][:, position].astype(np.int16)
        result[f"q{quantile}_factor_score_mean"] = quantile_statistics[
            "signal_means"
        ][:, position].astype(np.float32)
        result[f"q{quantile}_factor_score_median"] = quantile_statistics[
            "signal_medians"
        ][:, position].astype(np.float32)
        result[f"q{quantile}_return_mean"] = quantile_statistics[
            "return_means"
        ][:, position].astype(np.float32)
        result[f"q{quantile}_return_median"] = quantile_statistics[
            "return_medians"
        ][:, position].astype(np.float32)

    result["raw_spread"] = (
        result[f"q{QUANTILE_COUNT}_return_mean"]
        - result["q1_return_mean"]
    ).astype(np.float32)
    result["median_spread"] = (
        result[f"q{QUANTILE_COUNT}_return_median"]
        - result["q1_return_median"]
    ).astype(np.float32)
    return result


# -------------------------
# RESULT STREAM
def quantile_result_chunks(metadata, membership, forward_returns):
    index = membership.index
    membership_count = membership.sum(axis=1).to_numpy(dtype=np.int16)
    date_groups = np.repeat(
        np.arange(len(index), dtype=np.int32) * QUANTILE_COUNT,
        len(membership.columns),
    )

    for factor_number, factor_information in enumerate(
        metadata.itertuples(index=False),
        start=1,
    ):
        print(
            f"Factor {factor_number}/{len(metadata)}: "
            f"{factor_information.family} | {factor_information.variant}"
        )
        factor = load_factor_matrix(
            factor_information.family,
            factor_information.variant,
            membership,
        )
        signal, quantiles = prepare_factor_quantiles(factor, membership)

        for horizon in FORWARD_HORIZONS:
            yield build_quantile_result_chunk(
                factor_information,
                horizon,
                index,
                membership_count,
                date_groups,
                signal,
                quantiles,
                forward_returns[horizon],
            )


# -------------------------
# COMPLETE DAILY QUANTILE ANALYSIS
def run_quantile_analysis():
    prepare_selection_directories()
    metadata = load_factor_metadata()
    membership = load_membership()
    forward_returns = load_forward_return_matrices(
        FORWARD_HORIZONS,
        membership,
    )
    expected_rows = len(membership) * len(metadata) * len(FORWARD_HORIZONS)
    row_count, chunk_count = save_parquet_chunks(
        quantile_result_chunks(metadata, membership, forward_returns),
        DAILY_QUANTILE_RESULTS_PATH,
    )

    if row_count != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} quantile rows, created {row_count:,}"
        )

    eligible_dates = int(research_date_mask(membership.index).sum())
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start": membership.index.min().date().isoformat(),
        "data_end": membership.index.max().date().isoformat(),
        "research_start": RESEARCH_START_DATE,
        "research_end": RESEARCH_END_DATE,
        "trading_dates": len(membership),
        "research_eligible_dates": eligible_dates,
        "ticker_columns": len(membership.columns),
        "factor_variants": len(metadata),
        "forward_horizons": list(FORWARD_HORIZONS),
        "hypotheses": len(metadata) * len(FORWARD_HORIZONS),
        "quantiles": QUANTILE_COUNT,
        "signal_lag": SIGNAL_LAG,
        "minimum_assets": MIN_ASSETS,
        "factor_score_units": "daily cross-sectional z-score",
        "daily_relationship_metrics": [
            "spearman_ic",
            "pearson_correlation",
            "factor_beta",
        ],
        "quantile_statistics": [
            "count",
            "factor_score_mean",
            "factor_score_median",
            "return_mean",
            "return_median",
        ],
        "result_rows": row_count,
        "research_eligible_rows": (
            eligible_dates * len(metadata) * len(FORWARD_HORIZONS)
        ),
        "parquet_row_groups": chunk_count,
    }
    save_json(run_metadata, QUANTILE_RUN_METADATA_PATH)

    print("Daily quantile analysis is ready")
    print(f"Hypotheses: {run_metadata['hypotheses']}")
    print(f"Result rows: {row_count:,}")
    print(f"Results: {DAILY_QUANTILE_RESULTS_PATH}")

    return run_metadata


if __name__ == "__main__":
    run_quantile_analysis()
