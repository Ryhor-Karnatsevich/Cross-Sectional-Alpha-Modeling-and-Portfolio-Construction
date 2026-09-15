from datetime import datetime, timezone
from time import perf_counter

import pandas as pd

from factor_overlap import run_factor_overlap
from multiple_testing_research import compare_multiple_testing
from portfolio_engine import aggregate_calendar_phases, build_phase_paths
from portfolio_evaluation import evaluate_portfolios
from regime_research_current import run_regime_analysis
from research_config import (
    AVAILABILITY_PATH,
    BETA_NEUTRAL_OPTIONS,
    CALENDAR_PHASES_PER_FREQUENCY,
    FACTOR_METADATA_PATH,
    FACTOR_OVERLAP_PATH,
    MARKET_REGIMES_PATH,
    MEMBERSHIP_PATH,
    MULTIPLE_TESTING_RESULTS_PATH,
    MULTIPLE_TESTING_SUMMARY_PATH,
    PHASE_STABILITY_PATH,
    PORTFOLIO_METHODS,
    PORTFOLIO_STATISTICS_PATH,
    PRIMARY_TRANSACTION_COST_BPS,
    REBALANCE_FREQUENCIES,
    REGIME_RESULTS_PATH,
    RESEARCH_CACHE_MANIFEST_PATH,
    RESEARCH_REPORT_PATH,
    RESEARCH_RUN_METADATA_PATH,
    RETURNS_PATH,
    RISK_EXPOSURE_SUMMARY_PATH,
    RISK_FREE_RATE_PATH,
    SECTOR_EXPOSURE_STATUS_PATH,
    SECTOR_HISTORY_PATH,
    SELECTION_CARDS_PATH,
    SELECTION_EFFECTS_PATH,
    TRANSACTION_COST_BPS,
    WALK_FORWARD_PATHS_PATH,
    WALK_FORWARD_PERIODS_PATH,
    WALK_FORWARD_SUMMARY_PATH,
)
from research_data import (
    build_sector_matrix,
    load_candidate_factors,
    load_research_data,
    load_risk_free_daily,
    load_sector_history,
)
from research_report import build_research_report, create_research_figures
from research_storage import (
    input_signature,
    load_portfolio_cache,
    portfolio_cache_is_valid,
    prepare_research_directories,
    save_csv,
    save_json,
    save_parquet,
    save_portfolio_cache,
    save_text,
)
from risk_exposure import (
    build_market_regimes,
    build_market_return,
    build_trailing_betas,
)
from walk_forward_research import run_walk_forward


def run_multiple_testing_comparison():
    effect_tests = pd.read_csv(SELECTION_EFFECTS_PATH)
    comparison, summary = compare_multiple_testing(effect_tests)
    save_csv(comparison, MULTIPLE_TESTING_RESULTS_PATH)
    save_csv(summary, MULTIPLE_TESTING_SUMMARY_PATH)
    return comparison, summary


def cache_settings():
    return {
        "portfolio_methods": PORTFOLIO_METHODS,
        "beta_neutral_options": BETA_NEUTRAL_OPTIONS,
        "rebalance_frequencies": REBALANCE_FREQUENCIES,
        "calendar_phases_per_frequency": CALENDAR_PHASES_PER_FREQUENCY,
    }


def prepare_portfolio_paths(
    returns,
    membership,
    availability,
    factors,
    candidates,
    sector_matrix,
):
    input_paths = [
        RETURNS_PATH,
        MEMBERSHIP_PATH,
        AVAILABILITY_PATH,
        RISK_FREE_RATE_PATH,
        FACTOR_METADATA_PATH,
        SELECTION_CARDS_PATH,
        SELECTION_EFFECTS_PATH,
        SECTOR_HISTORY_PATH,
        *candidates["factor_path"].tolist(),
    ]
    signature, payload = input_signature(input_paths, cache_settings())
    if portfolio_cache_is_valid(signature):
        print("Reuse valid Research Layer portfolio cache")
        matrices, metadata = load_portfolio_cache()
        phase_stability = pd.read_csv(PHASE_STABILITY_PATH)
        return matrices, metadata, phase_stability, False

    market_return = build_market_return(returns, membership, availability)
    print("Build trailing point-in-time betas")
    betas = build_trailing_betas(returns, market_return)
    phase_matrices, phase_metadata = build_phase_paths(
        factors,
        candidates,
        returns,
        membership,
        availability,
        betas,
        sector_matrix,
    )
    matrices, metadata, phase_stability = aggregate_calendar_phases(
        phase_matrices,
        phase_metadata,
    )
    save_portfolio_cache(matrices, metadata, signature, payload)
    save_csv(phase_stability, PHASE_STABILITY_PATH)
    return matrices, metadata, phase_stability, True


def risk_summary(portfolio_statistics):
    columns = [
        "path_key",
        "factor_key",
        "method",
        "beta_neutral",
        "rebalance_days",
        "average_estimated_beta",
        "maximum_absolute_estimated_beta",
        "regression_beta",
        "average_gross_exposure",
        "average_net_exposure",
        "annualized_turnover",
        "unpriced_weight_days",
        "maximum_unpriced_weight",
        "average_maximum_sector_exposure",
        "maximum_sector_exposure",
        "average_unknown_sector_weight",
        "average_holding_count",
        "minimum_holding_count",
        "maximum_position_weight",
    ]
    return portfolio_statistics.loc[
        portfolio_statistics["transaction_cost_bps"].eq(
            PRIMARY_TRANSACTION_COST_BPS
        ),
        columns,
    ].copy()


def run_pipeline():
    start_time = perf_counter()
    prepare_research_directories()
    try:
        _, multiple_testing_summary = run_multiple_testing_comparison()
        returns, membership, availability = load_research_data()
        factors, candidates = load_candidate_factors(
            returns.index,
            returns.columns,
        )
        risk_free = load_risk_free_daily(returns.index)
        factor_overlap = run_factor_overlap(
            factors,
            membership,
            availability,
        )
        save_csv(factor_overlap, FACTOR_OVERLAP_PATH)
        sectors, sector_status_data = load_sector_history()
        sector_matrix = build_sector_matrix(
            sectors,
            returns.index,
            returns.columns,
        )
        sector_status = pd.DataFrame([sector_status_data])
        save_csv(sector_status, SECTOR_EXPOSURE_STATUS_PATH)

        market_return = build_market_return(returns, membership, availability)
        regimes = build_market_regimes(
            returns,
            membership,
            availability,
            market_return,
            risk_free,
        )
        save_parquet(regimes, MARKET_REGIMES_PATH)

        matrices, metadata, phase_stability, cache_rebuilt = (
            prepare_portfolio_paths(
                returns,
                membership,
                availability,
                factors,
                candidates,
                sector_matrix,
            )
        )
        statistics = evaluate_portfolios(
            matrices,
            metadata,
            market_return,
            risk_free["daily_rate"],
        )
        save_csv(statistics, PORTFOLIO_STATISTICS_PATH)
        risks = risk_summary(statistics)
        save_csv(risks, RISK_EXPOSURE_SUMMARY_PATH)

        walk_periods, walk_summary, walk_paths = run_walk_forward(
            matrices,
            metadata,
            risk_free["daily_rate"],
        )
        save_csv(walk_periods, WALK_FORWARD_PERIODS_PATH)
        save_csv(walk_summary, WALK_FORWARD_SUMMARY_PATH)
        save_parquet(walk_paths.astype("float32"), WALK_FORWARD_PATHS_PATH)

        regime_results = run_regime_analysis(
            walk_paths,
            regimes,
            walk_summary,
        )
        save_csv(regime_results, REGIME_RESULTS_PATH)
        figure_paths = create_research_figures(
            multiple_testing_summary,
            statistics,
            walk_summary,
            walk_paths,
            regime_results,
            factor_overlap,
        )
        report = build_research_report(
            candidates,
            multiple_testing_summary,
            statistics,
            phase_stability,
            walk_summary,
            risks,
            sector_status,
            regime_results,
            factor_overlap,
            figure_paths,
        )
        save_text(report, RESEARCH_REPORT_PATH)

        run_metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "portfolio_path_count": metadata["path_key"].nunique(),
            "portfolio_cache_rebuilt": cache_rebuilt,
            "transaction_cost_bps": list(TRANSACTION_COST_BPS),
            "primary_transaction_cost_bps": PRIMARY_TRANSACTION_COST_BPS,
            "walk_forward_period_rows": len(walk_periods),
            "complete_walk_forward_paths": len(walk_summary),
            "sector_exposure_status": sector_status_data["status"],
            "figure_count": len(figure_paths),
            "cache_manifest": RESEARCH_CACHE_MANIFEST_PATH,
        }
        save_json(run_metadata, RESEARCH_RUN_METADATA_PATH)
        print("Research Layer is ready")
        print(f"Portfolio paths: {run_metadata['portfolio_path_count']}")
        print(f"Report: {RESEARCH_REPORT_PATH}")
        return walk_summary
    finally:
        elapsed = perf_counter() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            "Research Layer pipeline time: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        )


if __name__ == "__main__":
    run_pipeline()
