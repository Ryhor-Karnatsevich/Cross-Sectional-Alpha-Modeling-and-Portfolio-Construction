from datetime import datetime, timezone

from factor_config import (
    APPLY_WINSORIZATION,
    FACTOR_RUN_METADATA_PATH,
    FACTOR_VARIANT_COUNT,
    FORWARD_HORIZONS,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    ROBUSTNESS_CONFIGS,
)
from factor_storage import (
    load_factor_inputs,
    prepare_factor_directories,
    save_factor_results,
)
from robustness import run_robustness, save_selected_signals
from sensitivity import run_sensitivity


# -------------------------
# COMPLETE FACTOR LAYER
def run_pipeline():
    print("Preparing Factor Layer directories...")
    prepare_factor_directories()

    print("Loading Data System matrices...")
    inputs = load_factor_inputs()

    sensitivity_results, daily_ic, metadata = run_sensitivity(inputs)
    robustness_results, selected_configs = run_robustness(
        daily_ic,
        metadata,
        inputs["prices"].index,
    )
    saved_matrices = save_selected_signals(
        selected_configs,
        inputs["prices"].index,
        inputs["prices"].columns,
    )

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start": inputs["prices"].index.min().date().isoformat(),
        "data_end": inputs["prices"].index.max().date().isoformat(),
        "research_start": RESEARCH_START_DATE,
        "research_end": RESEARCH_END_DATE,
        "trading_dates": len(inputs["prices"]),
        "ticker_columns": len(inputs["prices"].columns),
        "factor_families": metadata["family"].nunique(),
        "factor_variants": FACTOR_VARIANT_COUNT,
        "forward_horizons": list(FORWARD_HORIZONS),
        "hypotheses": len(metadata),
        "winsorization_applied": APPLY_WINSORIZATION,
        "robustness_windows": {
            layer: int(
                robustness_results.loc[
                    robustness_results["robustness_layer"] == layer,
                    "window",
                ].nunique()
            )
            for layer in ROBUSTNESS_CONFIGS
        },
        "selected_configurations": len(selected_configs),
        "selected_matrix_pairs": len(saved_matrices),
    }

    save_factor_results(
        sensitivity_results,
        robustness_results,
        selected_configs,
        run_metadata,
    )

    print("Factor Layer is ready")
    print(f"Factor variants: {FACTOR_VARIANT_COUNT}")
    print(f"Sensitivity hypotheses: {len(metadata)}")
    print(f"Selected configurations: {len(selected_configs)}")
    print(f"Run metadata: {FACTOR_RUN_METADATA_PATH}")

    return sensitivity_results, robustness_results, selected_configs


if __name__ == "__main__":
    run_pipeline()
