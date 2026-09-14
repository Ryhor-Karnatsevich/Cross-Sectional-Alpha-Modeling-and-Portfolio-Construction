import os


# -------------------------
# PATHS
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DATA_SYSTEM_PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Data_System",
    "Processed",
)
FACTOR_CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Factors_Layer",
    "Cache",
)
FACTOR_MATRIX_DIR = os.path.join(FACTOR_CACHE_DIR, "Factor_Matrices")
FORWARD_RETURN_MATRIX_DIR = os.path.join(
    FACTOR_CACHE_DIR,
    "Forward_Return_Matrices",
)
FACTOR_METADATA_PATH = os.path.join(
    FACTOR_CACHE_DIR,
    "factor_metadata.csv",
)
MEMBERSHIP_PATH = os.path.join(
    DATA_SYSTEM_PROCESSED_DIR,
    "membership.parquet",
)

SELECTION_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Factor_Selection_Layer",
)
SELECTION_CACHE_DIR = os.path.join(SELECTION_DATA_DIR, "Cache")
SELECTION_RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "Results",
    "Factor_Selection_Layer",
)
SELECTION_FIGURES_DIR = os.path.join(SELECTION_RESULTS_DIR, "Figures")
DAILY_QUANTILE_RESULTS_PATH = os.path.join(
    SELECTION_CACHE_DIR,
    "daily_quantile_results.parquet",
)
QUANTILE_RUN_METADATA_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "quantile_run_metadata.json",
)
HYPOTHESIS_CARDS_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "hypothesis_cards.csv",
)
EFFECT_TESTS_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "effect_tests.csv",
)
QUANTILE_CURVES_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "quantile_curves.csv",
)
TIME_STABILITY_PATH = os.path.join(
    SELECTION_CACHE_DIR,
    "time_stability.parquet",
)
SELECTION_REPORT_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "factor_selection_report.md",
)
SELECTION_RUN_METADATA_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "selection_run_metadata.json",
)
CLASSIFIER_VALIDATION_PATH = os.path.join(
    SELECTION_RESULTS_DIR,
    "classifier_validation.csv",
)


# -------------------------
# DAILY QUANTILE ANALYSIS
FORWARD_HORIZONS = (1, 5, 10, 21, 42, 63, 126, 252)
SIGNAL_LAG = 1
QUANTILE_COUNT = 10
MIN_ASSETS = 30
RESEARCH_START_DATE = "2010-01-01"
RESEARCH_END_DATE = None


# -------------------------
# CURRENT ANALYSIS SCOPE
# None analyzes all 56 factor configurations and 448 hypotheses.
ACTIVE_FACTOR_KEYS = None


# -------------------------
# HYPOTHESIS ANALYSIS
EFFECT_NAMES = (
    "spearman_ic",
    "q10_minus_q1",
    "q10_minus_middle",
    "middle_minus_q1",
    "edges_minus_middle",
)
MIDDLE_QUANTILES = (4, 5, 6, 7)
MIN_VALID_DATE_RATIO = 0.90
PATTERN_ABSOLUTE_TSTAT = 1.96
MONOTONIC_ABSOLUTE_RHO = 0.70
MONOTONIC_STEP_RATIO = 0.60
TAIL_DOMINANCE_RATIO = 1.25
MIN_MONTHLY_DIRECTION_RATE = 0.55
MIN_YEARLY_DIRECTION_RATE = 0.60
MULTIPLE_TESTING_ALPHA = 0.05
ROLLING_IC_WINDOW = 126
MAX_FACTOR_FIGURES = 10
