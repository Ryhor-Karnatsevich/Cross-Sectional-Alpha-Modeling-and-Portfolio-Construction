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
