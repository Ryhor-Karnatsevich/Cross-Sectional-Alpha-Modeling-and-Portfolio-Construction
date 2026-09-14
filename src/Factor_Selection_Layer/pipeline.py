import os
from datetime import datetime, timezone
from time import perf_counter

from classifier_validation import run_classifier_validation
from hypothesis_analysis import build_hypothesis_metrics
from pattern_classification import classify_hypotheses
from quantile_analysis import run_quantile_analysis
from selection_config import (
    ACTIVE_FACTOR_KEYS,
    DAILY_QUANTILE_RESULTS_PATH,
    EFFECT_TESTS_PATH,
    HYPOTHESIS_CARDS_PATH,
    QUANTILE_CURVES_PATH,
    SELECTION_REPORT_PATH,
    SELECTION_RUN_METADATA_PATH,
    TIME_STABILITY_PATH,
)
from selection_report import build_selection_report, create_selection_figures
from selection_storage import (
    prepare_selection_directories,
    save_csv,
    save_json,
    save_parquet,
    save_text,
)


# -------------------------
# QUANTILE DATA FOUNDATION
def prepare_quantile_data():
    if os.path.exists(DAILY_QUANTILE_RESULTS_PATH):
        print("Reuse existing daily_quantile_results.parquet")
        return False

    run_quantile_analysis()
    return True


# -------------------------
# HYPOTHESIS CLASSIFICATION
def run_factor_classification():
    classifier_validation = run_classifier_validation()
    cards, effect_tests, curves, time_stability, factor_count = (
        build_hypothesis_metrics()
    )
    cards, effect_tests = classify_hypotheses(
        cards,
        effect_tests,
        factor_count,
    )
    save_csv(cards, HYPOTHESIS_CARDS_PATH)
    save_csv(effect_tests, EFFECT_TESTS_PATH)
    save_csv(curves, QUANTILE_CURVES_PATH)
    save_parquet(time_stability, TIME_STABILITY_PATH)
    figure_paths = create_selection_figures(
        cards,
        curves,
        time_stability,
    )
    report = build_selection_report(cards, effect_tests, figure_paths)
    save_text(report, SELECTION_REPORT_PATH)
    return (
        cards,
        effect_tests,
        curves,
        time_stability,
        figure_paths,
        classifier_validation,
    )


# -------------------------
# COMPLETE FACTOR SELECTION LAYER
def run_pipeline():
    start_time = perf_counter()
    prepare_selection_directories()

    try:
        quantile_data_rebuilt = prepare_quantile_data()
        (
            cards,
            effect_tests,
            curves,
            time_stability,
            figure_paths,
            classifier_validation,
        ) = run_factor_classification()
        full_scope = bool(effect_tests["full_research_scope"].iloc[0])
        run_metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "active_factor_keys": (
                list(ACTIVE_FACTOR_KEYS)
                if ACTIVE_FACTOR_KEYS is not None
                else None
            ),
            "factor_configurations_analyzed": cards[
                "factor_key"
            ].nunique(),
            "hypotheses_analyzed": cards["hypothesis_key"].nunique(),
            "effect_tests_analyzed": len(effect_tests),
            "full_448_hypothesis_scope": full_scope,
            "quantile_data_rebuilt": quantile_data_rebuilt,
            "figures_created": len(figure_paths),
            "classifier_validation_passed": bool(
                classifier_validation["passed"].all()
            ),
            "pattern_counts": cards["pattern"].value_counts().to_dict(),
            "evidence_status_counts": cards[
                "evidence_status"
            ].value_counts().to_dict(),
        }
        save_json(run_metadata, SELECTION_RUN_METADATA_PATH)
        print("Factor classification is ready")
        print(
            "Factors analyzed: "
            f"{run_metadata['factor_configurations_analyzed']}"
        )
        print(f"Hypotheses analyzed: {run_metadata['hypotheses_analyzed']}")
        print(f"Cards: {HYPOTHESIS_CARDS_PATH}")
        print(f"Report: {SELECTION_REPORT_PATH}")
        return cards
    finally:
        elapsed_seconds = perf_counter() - start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            "Factor Selection pipeline time: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        )


if __name__ == "__main__":
    run_pipeline()
