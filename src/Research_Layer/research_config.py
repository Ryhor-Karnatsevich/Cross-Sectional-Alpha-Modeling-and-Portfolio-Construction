import os


# -------------------------
# PATHS
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
RESEARCH_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Research_Layer")
RESEARCH_CACHE_DIR = os.path.join(RESEARCH_DATA_DIR, "Cache")
RESEARCH_RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "Research_Layer")
RESEARCH_FIGURES_DIR = os.path.join(RESEARCH_RESULTS_DIR, "Figures")


# -------------------------
# PORTFOLIO EVALUATION
REBALANCE_STEP = 21
CALENDAR_PHASES = 21


# -------------------------
# OPTIONAL REPEATED-PERIOD IC RESEARCH
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
