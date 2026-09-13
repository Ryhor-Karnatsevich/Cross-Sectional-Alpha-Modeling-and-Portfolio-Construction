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


# -------------------------
# DAILY QUANTILE ANALYSIS
FORWARD_HORIZONS = (1, 5, 10, 21, 42, 63, 126, 252)
SIGNAL_LAG = 1
QUANTILE_COUNT = 5
MIN_ASSETS = 30
RESEARCH_START_DATE = "2010-01-01"
RESEARCH_END_DATE = None


# -------------------------
# OPTIONAL IC ANALYSIS
MIN_SELECTION_IC_OBSERVATIONS = 60
MIN_OOS_IC_OBSERVATIONS = 20


# -------------------------
# OPTIONAL ROBUSTNESS WINDOWS
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
        "horizons": FORWARD_HORIZONS,
    },
}
