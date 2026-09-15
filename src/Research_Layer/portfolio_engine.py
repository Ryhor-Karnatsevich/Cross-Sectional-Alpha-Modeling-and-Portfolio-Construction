import numpy as np
import pandas as pd

from portfolio_construction import (
    build_target_weights,
    build_target_weights_from_ranks,
    percentile_ranks,
)
from research_config import (
    BETA_NEUTRAL_OPTIONS,
    CALENDAR_PHASES_PER_FREQUENCY,
    MARKET_NEUTRAL_METHODS,
    PORTFOLIO_METHODS,
    PRIMARY_TRANSACTION_COST_BPS,
    REBALANCE_FREQUENCIES,
    RESEARCH_START_DATE,
    SIGNAL_LAG,
)


MATRIX_NAMES = (
    "gross_returns",
    "turnover",
    "beta_exposure",
    "gross_exposure",
    "net_exposure",
    "unpriced_weight",
    "max_sector_exposure",
    "unknown_sector_weight",
    "max_position",
    "holding_count",
)


def calendar_offsets(rebalance_days):
    count = min(int(rebalance_days), CALENDAR_PHASES_PER_FREQUENCY)
    return tuple(
        np.unique(
            np.linspace(0, int(rebalance_days) - 1, count, dtype=int)
        ).tolist()
    )


def path_key(factor_key, method, beta_neutral, rebalance_days, offset):
    beta_label = "beta_neutral" if beta_neutral else "unconstrained_beta"
    return (
        f"{factor_key}|{method}|{beta_label}|"
        f"r{rebalance_days}|p{offset}"
    )


def ensemble_key(factor_key, method, beta_neutral, rebalance_days):
    beta_label = "beta_neutral" if beta_neutral else "unconstrained_beta"
    return f"{factor_key}|{method}|{beta_label}|r{rebalance_days}|ensemble"


def path_specifications(candidates):
    rows = []
    for candidate in candidates.itertuples(index=False):
        for method in PORTFOLIO_METHODS:
            beta_options = (
                BETA_NEUTRAL_OPTIONS
                if method in MARKET_NEUTRAL_METHODS
                else (False,)
            )
            for beta_neutral in beta_options:
                for rebalance_days in REBALANCE_FREQUENCIES:
                    for offset in calendar_offsets(rebalance_days):
                        rows.append(
                            {
                                "path_key": path_key(
                                    candidate.factor_key,
                                    method,
                                    beta_neutral,
                                    rebalance_days,
                                    offset,
                                ),
                                "factor_key": candidate.factor_key,
                                "family": candidate.family,
                                "variant": candidate.variant,
                                "selection_horizon": int(
                                    candidate.selection_horizon
                                ),
                                "selection_evidence": (
                                    candidate.selection_evidence
                                ),
                                "selection_pattern": (
                                    candidate.selection_pattern
                                ),
                                "method": method,
                                "portfolio_type": (
                                    "market_neutral"
                                    if method in MARKET_NEUTRAL_METHODS
                                    else "long_only"
                                ),
                                "beta_neutral": bool(beta_neutral),
                                "rebalance_days": int(rebalance_days),
                                "calendar_offset": int(offset),
                            }
                        )
    return pd.DataFrame(rows)


def drift_weights(weights, asset_returns, portfolio_return):
    denominator = 1.0 + portfolio_return
    if not np.isfinite(denominator) or denominator <= 0:
        return np.zeros_like(weights)
    return weights * (1.0 + asset_returns) / denominator


def simulate_path(
    factor,
    returns,
    membership,
    availability,
    betas,
    sector_matrix,
    method,
    beta_neutral,
    rebalance_days,
    calendar_offset,
):
    factor = factor.shift(SIGNAL_LAG)
    index = returns.index
    columns = returns.columns
    factor_values = factor.to_numpy(dtype=float, na_value=np.nan)
    return_values = returns.to_numpy(dtype=float, na_value=np.nan)
    member_values = membership.to_numpy(dtype=bool)
    available_values = availability.to_numpy(dtype=bool)
    beta_values = betas.to_numpy(dtype=float, na_value=np.nan)
    sector_values = (
        sector_matrix.to_numpy(dtype=object)
        if sector_matrix is not None
        else None
    )
    start_position = int(index.searchsorted(RESEARCH_START_DATE))
    asset_count = len(columns)
    weights = np.zeros(asset_count, dtype=float)

    output = {
        name: np.zeros(len(index), dtype=float)
        for name in MATRIX_NAMES
    }
    successful_rebalances = 0
    skipped_rebalances = 0
    beta_neutral_attempts = 0
    beta_neutral_successes = 0

    for position in range(start_position, len(index)):
        # Weights enter date t before the t-1 -> t return is earned. The signal
        # is already shifted, so a rebalance at t only uses factor[t-1].
        previous = weights.copy()
        target = previous.copy()
        target[~member_values[position]] = 0.0

        first_rebalance = start_position + int(calendar_offset)
        rebalance = (
            position >= first_rebalance
            and (position - first_rebalance) % int(rebalance_days) == 0
        )
        if rebalance:
            eligible = (
                member_values[position]
                & available_values[position]
                & np.isfinite(factor_values[position])
            )
            scores = pd.Series(factor_values[position], index=columns)
            eligible_series = pd.Series(eligible, index=columns)
            beta_series = pd.Series(beta_values[position], index=columns)
            built, beta_applied = build_target_weights(
                scores,
                eligible_series,
                method,
                beta_series,
                beta_neutral,
            )
            if beta_neutral:
                beta_neutral_attempts += 1
                beta_neutral_successes += int(beta_applied)
            if built.empty:
                target = np.zeros(asset_count, dtype=float)
                skipped_rebalances += 1
            else:
                target = built.reindex(columns, fill_value=0.0).to_numpy()
                successful_rebalances += 1

        output["turnover"][position] = np.abs(target - previous).sum()
        raw_asset_returns = return_values[position]
        missing_held = ~np.isfinite(raw_asset_returns) & (np.abs(target) > 0)
        output["unpriced_weight"][position] = np.abs(
            target[missing_held]
        ).sum()
        asset_returns = np.where(
            np.isfinite(raw_asset_returns),
            raw_asset_returns,
            0.0,
        )
        portfolio_return = float(np.dot(target, asset_returns))
        output["gross_returns"][position] = portfolio_return
        finite_beta = np.where(
            np.isfinite(beta_values[position]),
            beta_values[position],
            0.0,
        )
        output["beta_exposure"][position] = np.dot(target, finite_beta)
        output["gross_exposure"][position] = np.abs(target).sum()
        output["net_exposure"][position] = target.sum()
        output["max_position"][position] = (
            np.abs(target).max() if np.any(target) else 0.0
        )
        output["holding_count"][position] = np.count_nonzero(target)
        if sector_values is not None:
            labels = sector_values[position]
            known = pd.notna(labels) & (np.abs(target) > 0)
            unknown = pd.isna(labels) & (np.abs(target) > 0)
            output["unknown_sector_weight"][position] = np.abs(
                target[unknown]
            ).sum()
            if known.any():
                sector_exposure = pd.Series(target[known]).groupby(
                    labels[known]
                ).sum()
                output["max_sector_exposure"][position] = (
                    sector_exposure.abs().max()
                )
        weights = drift_weights(target, asset_returns, portfolio_return)

    diagnostics = {
        "successful_rebalances": successful_rebalances,
        "skipped_rebalances": skipped_rebalances,
        "beta_neutral_attempts": beta_neutral_attempts,
        "beta_neutral_success_rate": (
            beta_neutral_successes / beta_neutral_attempts
            if beta_neutral_attempts
            else np.nan
        ),
    }
    series = {
        name: pd.Series(values, index=index)
        for name, values in output.items()
    }
    return series, diagnostics


def build_phase_paths(
    factors,
    candidates,
    returns,
    membership,
    availability,
    betas,
    sector_matrix=None,
):
    specifications = path_specifications(candidates)
    outputs = {name: {} for name in MATRIX_NAMES}
    metadata_rows = []

    for factor_number, (factor_key_value, factor_specs) in enumerate(
        specifications.groupby("factor_key", sort=False),
        start=1,
    ):
        print(
            f"Portfolio factor {factor_number}/{specifications['factor_key'].nunique()}: "
            f"{factor_key_value}"
        )
        factor_outputs, factor_metadata = simulate_factor_paths(
            factors[factor_key_value],
            factor_specs.reset_index(drop=True),
            returns,
            membership,
            availability,
            betas,
            sector_matrix,
        )
        for name in MATRIX_NAMES:
            for path in factor_outputs[name].columns:
                outputs[name][path] = factor_outputs[name][path]
        metadata_rows.extend(factor_metadata.to_dict("records"))

    matrices = {
        name: pd.DataFrame(columns)
        for name, columns in outputs.items()
    }
    return matrices, pd.DataFrame(metadata_rows)


def simulate_factor_paths(
    factor,
    specifications,
    returns,
    membership,
    availability,
    betas,
    sector_matrix=None,
):
    index = returns.index
    columns = returns.columns
    factor_values = factor.shift(SIGNAL_LAG).to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    return_values = returns.to_numpy(dtype=float, na_value=np.nan)
    member_values = membership.to_numpy(dtype=bool)
    available_values = availability.to_numpy(dtype=bool)
    beta_values = betas.to_numpy(dtype=float, na_value=np.nan)
    sector_values = (
        sector_matrix.to_numpy(dtype=object)
        if sector_matrix is not None
        else None
    )
    start_position = int(index.searchsorted(RESEARCH_START_DATE))
    path_count = len(specifications)
    asset_count = len(columns)
    weights = np.zeros((path_count, asset_count), dtype=float)
    output = {
        name: np.zeros((len(index), path_count), dtype=float)
        for name in MATRIX_NAMES
    }
    successful = np.zeros(path_count, dtype=int)
    skipped = np.zeros(path_count, dtype=int)
    beta_attempts = np.zeros(path_count, dtype=int)
    beta_successes = np.zeros(path_count, dtype=int)
    frequencies = specifications["rebalance_days"].to_numpy(dtype=int)
    offsets = specifications["calendar_offset"].to_numpy(dtype=int)
    methods = specifications["method"].to_numpy(dtype=object)
    beta_requests = specifications["beta_neutral"].to_numpy(dtype=bool)
    implementation_keys = list(dict.fromkeys(zip(methods, beta_requests)))

    for position in range(start_position, len(index)):
        previous = weights.copy()
        target = previous.copy()
        target[:, ~member_values[position]] = 0.0

        relative_position = position - start_position
        due = (
            relative_position >= offsets
        ) & ((relative_position - offsets) % frequencies == 0)

        if due.any():
            eligible = (
                member_values[position]
                & available_values[position]
                & np.isfinite(factor_values[position])
            )
            scores = pd.Series(factor_values[position], index=columns)
            ranks = percentile_ranks(
                scores,
                pd.Series(eligible, index=columns),
            )
            beta_series = pd.Series(beta_values[position], index=columns)

            for method, beta_request in implementation_keys:
                paths = due & (methods == method) & (
                    beta_requests == beta_request
                )
                if not paths.any():
                    continue
                built, beta_applied = build_target_weights_from_ranks(
                    ranks,
                    method,
                    beta_series,
                    bool(beta_request),
                )
                if beta_request:
                    beta_attempts[paths] += 1
                    beta_successes[paths] += int(beta_applied)
                if built.empty:
                    target[paths] = 0.0
                    skipped[paths] += 1
                else:
                    target[paths] = built.reindex(
                        columns,
                        fill_value=0.0,
                    ).to_numpy()
                    successful[paths] += 1

        output["turnover"][position] = np.abs(target - previous).sum(axis=1)
        raw_returns = return_values[position]
        finite_returns = np.isfinite(raw_returns)
        output["unpriced_weight"][position] = (
            np.abs(target[:, ~finite_returns]).sum(axis=1)
            if (~finite_returns).any()
            else 0.0
        )
        asset_returns = np.where(finite_returns, raw_returns, 0.0)
        portfolio_returns = target @ asset_returns
        output["gross_returns"][position] = portfolio_returns
        finite_beta = np.where(
            np.isfinite(beta_values[position]),
            beta_values[position],
            0.0,
        )
        output["beta_exposure"][position] = target @ finite_beta
        output["gross_exposure"][position] = np.abs(target).sum(axis=1)
        output["net_exposure"][position] = target.sum(axis=1)
        output["max_position"][position] = np.abs(target).max(axis=1)
        output["holding_count"][position] = np.count_nonzero(target, axis=1)

        if sector_values is not None:
            labels = sector_values[position]
            known_labels = pd.Index(labels[pd.notna(labels)]).unique()
            unknown = pd.isna(labels)
            output["unknown_sector_weight"][position] = np.abs(
                target[:, unknown]
            ).sum(axis=1)
            sector_exposures = np.zeros(path_count, dtype=float)
            for label in known_labels:
                sector_mask = labels == label
                sector_exposures = np.maximum(
                    sector_exposures,
                    np.abs(target[:, sector_mask].sum(axis=1)),
                )
            output["max_sector_exposure"][position] = sector_exposures

        denominators = 1.0 + portfolio_returns
        safe = np.isfinite(denominators) & (denominators > 0)
        weights = np.zeros_like(target)
        weights[safe] = (
            target[safe]
            * (1.0 + asset_returns)[None, :]
            / denominators[safe, None]
        )

    path_names = specifications["path_key"].tolist()
    matrices = {
        name: pd.DataFrame(values, index=index, columns=path_names)
        for name, values in output.items()
    }
    metadata = specifications.copy()
    metadata["successful_rebalances"] = successful
    metadata["skipped_rebalances"] = skipped
    metadata["beta_neutral_attempts"] = beta_attempts
    metadata["beta_neutral_success_rate"] = np.divide(
        beta_successes,
        beta_attempts,
        out=np.full(path_count, np.nan),
        where=beta_attempts > 0,
    )
    return matrices, metadata


def simple_path_statistics(gross, turnover):
    gross = gross.loc[RESEARCH_START_DATE:]
    turnover = turnover.reindex(gross.index)
    cost = turnover * PRIMARY_TRANSACTION_COST_BPS / 10_000
    net = (1 + gross) * (1 - cost) - 1
    net = net.dropna()
    if len(net) < 2:
        return np.nan, np.nan
    volatility = net.std() * np.sqrt(252)
    sharpe = net.mean() * 252 / volatility if volatility > 0 else np.nan
    wealth = (1 + net).prod()
    annual_return = (
        wealth ** (252 / len(net)) - 1 if wealth > 0 else np.nan
    )
    return annual_return, sharpe


def aggregate_calendar_phases(phase_matrices, phase_metadata):
    group_columns = [
        "factor_key",
        "family",
        "variant",
        "selection_horizon",
        "selection_evidence",
        "selection_pattern",
        "method",
        "portfolio_type",
        "beta_neutral",
        "rebalance_days",
    ]
    ensemble_outputs = {name: {} for name in MATRIX_NAMES}
    ensemble_rows = []
    stability_rows = []

    for key, group in phase_metadata.groupby(group_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key))
        columns = group["path_key"].tolist()
        name = ensemble_key(
            identity["factor_key"],
            identity["method"],
            bool(identity["beta_neutral"]),
            int(identity["rebalance_days"]),
        )
        for matrix_name in MATRIX_NAMES:
            ensemble_outputs[matrix_name][name] = phase_matrices[
                matrix_name
            ][columns].mean(axis=1)

        phase_stats = []
        for phase_path in columns:
            annual_return, sharpe = simple_path_statistics(
                phase_matrices["gross_returns"][phase_path],
                phase_matrices["turnover"][phase_path],
            )
            phase_stats.append((annual_return, sharpe))
        phase_stats = np.asarray(phase_stats, dtype=float)
        stability_rows.append(
            {
                "path_key": name,
                "phase_count": len(columns),
                "phase_annual_return_mean": np.nanmean(phase_stats[:, 0]),
                "phase_annual_return_min": np.nanmin(phase_stats[:, 0]),
                "phase_annual_return_max": np.nanmax(phase_stats[:, 0]),
                "phase_annual_return_std": np.nanstd(phase_stats[:, 0]),
                "phase_sharpe_mean": np.nanmean(phase_stats[:, 1]),
                "phase_sharpe_min": np.nanmin(phase_stats[:, 1]),
                "phase_sharpe_max": np.nanmax(phase_stats[:, 1]),
                "phase_sharpe_std": np.nanstd(phase_stats[:, 1]),
            }
        )
        ensemble_rows.append(
            {
                "path_key": name,
                **identity,
                "calendar_offset": "ensemble",
                "phase_count": len(columns),
                "successful_rebalances": group[
                    "successful_rebalances"
                ].mean(),
                "skipped_rebalances": group["skipped_rebalances"].mean(),
                "beta_neutral_attempts": group[
                    "beta_neutral_attempts"
                ].mean(),
                "beta_neutral_success_rate": group[
                    "beta_neutral_success_rate"
                ].mean(),
            }
        )

    matrices = {
        name: pd.DataFrame(columns)
        for name, columns in ensemble_outputs.items()
    }
    return (
        matrices,
        pd.DataFrame(ensemble_rows),
        pd.DataFrame(stability_rows),
    )
