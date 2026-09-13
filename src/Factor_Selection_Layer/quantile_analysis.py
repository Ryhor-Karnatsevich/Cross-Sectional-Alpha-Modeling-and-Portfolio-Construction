from datetime import datetime, timezone

import numpy as np
import pandas as pd

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
    return quantiles


# -------------------------
# QUANTILE RETURNS
def aggregate_quantile_returns(
    quantiles,
    forward_returns,
    date_groups=None,
):
    return_values = forward_returns.to_numpy(dtype="float32", na_value=np.nan)
    date_count, asset_count = quantiles.shape

    if return_values.shape != quantiles.shape:
        raise ValueError("Factor quantiles and forward returns must align")

    flat_quantiles = quantiles.ravel()
    flat_returns = return_values.ravel()
    valid = (flat_quantiles > 0) & np.isfinite(flat_returns)
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
    sums = np.bincount(
        groups,
        weights=flat_returns[valid],
        minlength=group_count,
    ).reshape(date_count, QUANTILE_COUNT)
    means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)

    signal_asset_count = (quantiles > 0).sum(axis=1)
    return_asset_count = counts.sum(axis=1)
    enough_assets = return_asset_count >= MIN_ASSETS
    means[~enough_assets] = np.nan

    return means, counts, signal_asset_count, return_asset_count


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
    quantiles,
    forward_returns,
):
    means, counts, signal_count, return_count = aggregate_quantile_returns(
        quantiles,
        forward_returns,
        date_groups,
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
            "signal_asset_count": signal_count.astype(np.int16),
            "return_asset_count": return_count.astype(np.int16),
        }
    )

    for quantile in range(1, QUANTILE_COUNT + 1):
        result[f"q{quantile}_count"] = counts[:, quantile - 1].astype(
            np.int16
        )
        result[f"q{quantile}_return"] = means[:, quantile - 1].astype(
            np.float32
        )

    result["raw_spread"] = (
        result[f"q{QUANTILE_COUNT}_return"] - result["q1_return"]
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
        quantiles = prepare_factor_quantiles(factor, membership)

        for horizon in FORWARD_HORIZONS:
            yield build_quantile_result_chunk(
                factor_information,
                horizon,
                index,
                membership_count,
                date_groups,
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
