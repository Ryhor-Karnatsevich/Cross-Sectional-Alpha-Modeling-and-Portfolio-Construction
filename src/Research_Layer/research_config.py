import os


# -------------------------
# PATHS
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_SYSTEM_DIR = os.path.join(PROJECT_ROOT, "Data", "Data_System")
DATA_SYSTEM_RAW_DIR = os.path.join(DATA_SYSTEM_DIR, "Raw")
DATA_SYSTEM_PROCESSED_DIR = os.path.join(DATA_SYSTEM_DIR, "Processed")
FACTOR_CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Factors_Layer",
    "Cache",
)
FACTOR_MATRIX_DIR = os.path.join(FACTOR_CACHE_DIR, "Factor_Matrices")
FACTOR_METADATA_PATH = os.path.join(
    FACTOR_CACHE_DIR,
    "factor_metadata.csv",
)
SELECTION_RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "Results",
    "Factor_Selection_Layer",
)

RESEARCH_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Research_Layer")
RESEARCH_CACHE_DIR = os.path.join(RESEARCH_DATA_DIR, "Cache")
RESEARCH_RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "Research_Layer")
RESEARCH_FIGURES_DIR = os.path.join(RESEARCH_RESULTS_DIR, "Figures")

RETURNS_PATH = os.path.join(DATA_SYSTEM_PROCESSED_DIR, "returns.parquet")
AVAILABILITY_PATH = os.path.join(
    DATA_SYSTEM_PROCESSED_DIR,
    "availability.parquet",
)
MEMBERSHIP_PATH = os.path.join(
    DATA_SYSTEM_PROCESSED_DIR,
    "membership.parquet",
)
RISK_FREE_RATE_PATH = os.path.join(
    DATA_SYSTEM_RAW_DIR,
    "dgs3mo.parquet",
)

# Optional point-in-time file. Required columns are ticker, sector, start_date
# and end_date. Current sectors are not silently applied to historical stocks.
SECTOR_HISTORY_PATH = os.path.join(
    DATA_SYSTEM_RAW_DIR,
    "sector_history.csv",
)

SELECTION_CARDS_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "hypothesis_cards.csv",
)
SELECTION_EFFECTS_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "effect_tests.csv",
)

MULTIPLE_TESTING_RESULTS_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "multiple_testing_comparison.csv",
)
MULTIPLE_TESTING_SUMMARY_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "multiple_testing_summary.csv",
)
FACTOR_OVERLAP_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "factor_overlap.csv",
)
PORTFOLIO_PATHS_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_paths.parquet",
)
PORTFOLIO_TURNOVER_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_turnover.parquet",
)
PORTFOLIO_BETA_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_beta_exposure.parquet",
)
PORTFOLIO_GROSS_EXPOSURE_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_gross_exposure.parquet",
)
PORTFOLIO_NET_EXPOSURE_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_net_exposure.parquet",
)
PORTFOLIO_UNPRICED_WEIGHT_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_unpriced_weight.parquet",
)
PORTFOLIO_MAX_SECTOR_EXPOSURE_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_max_sector_exposure.parquet",
)
PORTFOLIO_UNKNOWN_SECTOR_WEIGHT_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_unknown_sector_weight.parquet",
)
PORTFOLIO_MAX_POSITION_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_max_position.parquet",
)
PORTFOLIO_HOLDING_COUNT_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "portfolio_holding_count.parquet",
)
PORTFOLIO_METADATA_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "portfolio_metadata.csv",
)
PORTFOLIO_STATISTICS_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "portfolio_statistics.csv",
)
PHASE_STABILITY_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "calendar_phase_stability.csv",
)
WALK_FORWARD_PERIODS_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "walk_forward_periods.csv",
)
WALK_FORWARD_SUMMARY_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "walk_forward_summary.csv",
)
WALK_FORWARD_PATHS_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "walk_forward_oos_paths.parquet",
)
RISK_EXPOSURE_SUMMARY_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "risk_exposure_summary.csv",
)
SECTOR_EXPOSURE_STATUS_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "sector_exposure_status.csv",
)
MARKET_REGIMES_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "market_regimes.parquet",
)
REGIME_RESULTS_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "regime_performance.csv",
)
RESEARCH_REPORT_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "research_report.md",
)
RESEARCH_RUN_METADATA_PATH = os.path.join(
    RESEARCH_RESULTS_DIR,
    "research_run_metadata.json",
)
RESEARCH_CACHE_MANIFEST_PATH = os.path.join(
    RESEARCH_CACHE_DIR,
    "research_cache_manifest.json",
)


# -------------------------
# FROZEN RESEARCH CANDIDATES
# These are leads from the completed Selection Layer. They are not called
# validated alpha and are not expanded automatically after seeing Layer 4.
RESEARCH_CANDIDATES = (
    {
        "factor_key": (
            "volatility_scaled_momentum|vsmom_12m_1m_vol60"
        ),
        "selection_horizon": 1,
        "selection_evidence": "global_fdr_rank_signal",
    },
    {
        "factor_key": "short_term_reversal|reversal_5d",
        "selection_horizon": 5,
        "selection_evidence": "upper_tail_exploratory_lead",
    },
    {
        "factor_key": "liquidity_change|liq_20d_252d",
        "selection_horizon": 252,
        "selection_evidence": "upper_tail_exploratory_lead",
    },
    {
        "factor_key": "low_volatility|60d",
        "selection_horizon": 63,
        "selection_evidence": "lower_tail_exploratory_lead",
    },
)


# -------------------------
# MULTIPLE TESTING AUDIT
MULTIPLE_TESTING_ALPHA = 0.05
MULTIPLE_TESTING_GROUP_COLUMNS = ("family", "effect")


# -------------------------
# PORTFOLIO CONSTRUCTION
RESEARCH_START_DATE = "2010-01-01"
SIGNAL_LAG = 1
MIN_ASSETS = 30
MIN_NAMES_PER_LEG = 5
GROSS_EXPOSURE = 2.0
LONG_ONLY_EXPOSURE = 1.0
TAIL_FRACTION = 0.10
MIDDLE_LOWER = 0.30
MIDDLE_UPPER = 0.70

PORTFOLIO_METHODS = (
    "continuous_high_minus_low",
    "q10_minus_q1",
    "q10_minus_middle",
    "middle_minus_q1",
    "long_q10",
    "long_q1",
)
MARKET_NEUTRAL_METHODS = (
    "continuous_high_minus_low",
    "q10_minus_q1",
    "q10_minus_middle",
    "middle_minus_q1",
)
BETA_NEUTRAL_OPTIONS = (False, True)
REBALANCE_FREQUENCIES = (1, 5, 21, 63)
CALENDAR_PHASES_PER_FREQUENCY = 3


# -------------------------
# TRANSACTION COSTS AND RISK
TRANSACTION_COST_BPS = (0, 5, 10, 25)
PRIMARY_TRANSACTION_COST_BPS = 10
ANNUALIZATION_FACTOR = 252
BETA_LOOKBACK = 252
BETA_MIN_OBSERVATIONS = 126
REGRESSION_HAC_LAGS = 5


# -------------------------
# WALK-FORWARD
# Candidates were discovered on the complete history, so these results are a
# post-selection historical simulation, not a genuinely untouched final test.
WALK_FORWARD_SCHEMES = {
    "short": {
        "train_months": 18,
        "oos_months": 6,
        "step_months": 6,
    },
    "long": {
        "train_months": 48,
        "oos_months": 12,
        "step_months": 12,
    },
}
MIN_TRAIN_OBSERVATIONS = 500
MIN_SHORT_TRAIN_OBSERVATIONS = 300
MIN_OOS_OBSERVATION_RATIO = 0.75


# -------------------------
# MARKET REGIMES
REGIME_VOLATILITY_WINDOW = 63
REGIME_DISPERSION_WINDOW = 21
REGIME_CORRELATION_WINDOW = 63
REGIME_EXPANDING_MIN_OBSERVATIONS = 252
HIGH_RISK_FREE_RATE_PCT = 2.0


# -------------------------
# OPTIONAL LEGACY REPEATED-PERIOD IC RESEARCH
# Kept only so the old scripts remain readable. The current Layer 4 pipeline
# does not call the legacy pipeline.
REBALANCE_STEP = 21
CALENDAR_PHASES = 21
MIN_SELECTION_IC_OBSERVATIONS = 60
MIN_OOS_IC_OBSERVATIONS = 20
ROBUSTNESS_CONFIGS = {
    "short": {
        "selection_months": 18,
        "oos_months": 6,
        "step_months": 6,
        "horizons": (1, 5, 10, 21, 42, 63, 126),
    },
    "long": {
        "selection_years": 4,
        "oos_years": 1,
        "step_years": 1,
        "horizons": (1, 5, 10, 21, 42, 63, 126, 252),
    },
}
