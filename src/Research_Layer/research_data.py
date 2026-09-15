import os

import numpy as np
import pandas as pd

from research_config import (
    AVAILABILITY_PATH,
    FACTOR_MATRIX_DIR,
    FACTOR_METADATA_PATH,
    MEMBERSHIP_PATH,
    RESEARCH_CANDIDATES,
    RETURNS_PATH,
    RISK_FREE_RATE_PATH,
    SECTOR_HISTORY_PATH,
    SELECTION_CARDS_PATH,
)


def read_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def align_boolean_matrix(frame, index, columns):
    return (
        frame.reindex(index=index, columns=columns)
        .fillna(False)
        .astype(bool)
    )


def load_research_data():
    # Keep the Data System warm-up history. Portfolio results still begin at
    # RESEARCH_START_DATE inside the portfolio and walk-forward engines.
    returns = read_matrix(RETURNS_PATH)
    columns = returns.columns
    index = returns.index
    membership = align_boolean_matrix(
        read_matrix(MEMBERSHIP_PATH),
        index,
        columns,
    )
    availability = align_boolean_matrix(
        read_matrix(AVAILABILITY_PATH),
        index,
        columns,
    )
    return returns, membership, availability


def load_risk_free_daily(index):
    rate = read_matrix(RISK_FREE_RATE_PATH)
    if "annual_rate_pct" not in rate.columns:
        raise ValueError("Risk-free data must contain annual_rate_pct")
    annual = rate["annual_rate_pct"].reindex(index).ffill()
    daily = np.expm1(np.log1p(annual / 100.0) / 252.0)
    return pd.DataFrame(
        {
            "annual_rate_pct": annual,
            "daily_rate": daily,
        },
        index=index,
    )


def candidate_table():
    metadata = pd.read_csv(FACTOR_METADATA_PATH)
    cards = pd.read_csv(SELECTION_CARDS_PATH)
    rows = []

    for candidate in RESEARCH_CANDIDATES:
        factor_key = candidate["factor_key"]
        horizon = int(candidate["selection_horizon"])
        factor_row = metadata.loc[metadata["key"].eq(factor_key)]
        if len(factor_row) != 1:
            raise ValueError(f"Factor metadata is not unique: {factor_key}")
        card = cards.loc[
            cards["factor_key"].eq(factor_key)
            & cards["horizon_days"].eq(horizon)
        ]
        if len(card) != 1:
            raise ValueError(
                f"Selection card is not unique: {factor_key} | h{horizon}"
            )

        factor_row = factor_row.iloc[0]
        card = card.iloc[0]
        path = os.path.abspath(factor_row["path"])
        allowed_root = os.path.abspath(FACTOR_MATRIX_DIR)
        if os.path.commonpath([path, allowed_root]) != allowed_root:
            raise ValueError(f"Factor path is outside Factor Layer: {path}")

        rows.append(
            {
                **candidate,
                "family": factor_row["family"],
                "variant": factor_row["variant"],
                "factor_path": path,
                "selection_pattern": card["pattern"],
                "selection_pattern_direction": card["pattern_direction"],
                "selection_status": card["evidence_status"],
                "selection_ic": card["spearman_ic_mean"],
                "selection_ic_hac_tstat": card["spearman_ic_hac_tstat"],
                "selection_primary_effect": card["primary_effect"],
                "selection_primary_effect_mean": card["primary_effect_mean"],
                "selection_primary_effect_hac_tstat": card[
                    "primary_effect_hac_tstat"
                ],
            }
        )
    return pd.DataFrame(rows)


def load_candidate_factors(index, columns):
    candidates = candidate_table()
    factors = {}
    for row in candidates.itertuples(index=False):
        factor = read_matrix(row.factor_path)
        factors[row.factor_key] = factor.reindex(index=index, columns=columns)
    return factors, candidates


def load_sector_history():
    if not os.path.exists(SECTOR_HISTORY_PATH):
        return None, {
            "status": "NOT_AVAILABLE",
            "reason": (
                "Data System has no point-in-time sector_history.csv. "
                "Historical sector exposure was not estimated."
            ),
        }

    sectors = pd.read_csv(SECTOR_HISTORY_PATH)
    required = {"ticker", "sector", "start_date", "end_date"}
    missing = required.difference(sectors.columns)
    if missing:
        return None, {
            "status": "FAIL",
            "reason": "Missing columns: " + ", ".join(sorted(missing)),
        }
    sectors["start_date"] = pd.to_datetime(sectors["start_date"])
    sectors["end_date"] = pd.to_datetime(sectors["end_date"])
    return sectors, {
        "status": "AVAILABLE",
        "reason": "Point-in-time sector history loaded.",
    }


def build_sector_matrix(sectors, index, columns):
    if sectors is None:
        return None
    result = pd.DataFrame(pd.NA, index=index, columns=columns, dtype="object")
    column_set = set(columns)
    for row in sectors.itertuples(index=False):
        if row.ticker not in column_set:
            continue
        dates = (index >= row.start_date) & (index <= row.end_date)
        result.loc[dates, row.ticker] = row.sector
    return result
