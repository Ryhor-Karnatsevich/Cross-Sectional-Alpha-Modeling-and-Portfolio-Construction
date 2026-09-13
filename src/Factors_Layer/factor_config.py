import os


# -------------------------
# PATHS
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
FACTOR_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Factors_Layer")
FACTOR_CACHE_DIR = os.path.join(FACTOR_DATA_DIR, "Cache")
FACTOR_RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "Factors_Layer")
FACTOR_FIGURES_DIR = os.path.join(FACTOR_RESULTS_DIR, "Figures")
FACTOR_MATRIX_CACHE_DIR = os.path.join(FACTOR_CACHE_DIR, "Factor_Matrices")
FORWARD_RETURN_MATRIX_CACHE_DIR = os.path.join(
    FACTOR_CACHE_DIR,
    "Forward_Return_Matrices",
)
FACTOR_METADATA_CACHE_PATH = os.path.join(
    FACTOR_CACHE_DIR,
    "factor_metadata.csv",
)
FACTOR_CACHE_MANIFEST_PATH = os.path.join(
    FACTOR_CACHE_DIR,
    "cache_manifest.json",
)
FACTOR_RUN_METADATA_PATH = os.path.join(
    FACTOR_RESULTS_DIR,
    "factor_run_metadata.json",
)


# -------------------------
# FACTOR PREPARATION
MIN_OBSERVATION_RATIO = 0.80
APPLY_WINSORIZATION = False
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
ANNUALIZATION_FACTOR = 252

# -------------------------
# FORWARD RETURNS
FORWARD_HORIZONS = (1, 5, 10, 21, 42, 63, 126, 252)


#----------------------------------------------------------------------------------------------------
# MOMENTUM
MOMENTUM_CONFIGS = [
    {"variant": "3m-0m", "window": 63, "skip": 0},
    {"variant": "3m-1m", "window": 63, "skip": 21},
    {"variant": "6m-0m", "window": 126, "skip": 0},
    {"variant": "6m-1m", "window": 126, "skip": 21},
    {"variant": "6m-2m", "window": 126, "skip": 42},
    {"variant": "9m-0m", "window": 189, "skip": 0},
    {"variant": "9m-1m", "window": 189, "skip": 21},
    {"variant": "9m-2m", "window": 189, "skip": 42},
    {"variant": "12m-0m", "window": 252, "skip": 0},
    {"variant": "12m-1m", "window": 252, "skip": 21},
    {"variant": "12m-2m", "window": 252, "skip": 42},
    {"variant": "18m-0m", "window": 378, "skip": 0},
    {"variant": "18m-1m", "window": 378, "skip": 21},
    {"variant": "18m-2m", "window": 378, "skip": 42},
    {"variant": "24m-0m", "window": 504, "skip": 0},
    {"variant": "24m-1m", "window": 504, "skip": 21},
    {"variant": "24m-2m", "window": 504, "skip": 42},
]

# LOW VOLATILITY
LOW_VOLATILITY_CONFIGS = [
    {"variant": "20d", "window": 20},
    {"variant": "40d", "window": 40},
    {"variant": "60d", "window": 60},
    {"variant": "90d", "window": 90},
    {"variant": "120d", "window": 120},
    {"variant": "180d", "window": 180},
    {"variant": "252d", "window": 252},
]

# TREND
TREND_CONFIGS = [
    {"variant": "SMA20", "window": 20},
    {"variant": "SMA50", "window": 50},
    {"variant": "SMA100", "window": 100},
    {"variant": "SMA150", "window": 150},
    {"variant": "SMA200", "window": 200},
    {"variant": "SMA250", "window": 250},
]

# SHORT-TERM REVERSAL
SHORT_TERM_REVERSAL_CONFIGS = [
    {"variant": "reversal_5d", "window": 5},
    {"variant": "reversal_10d", "window": 10},
    {"variant": "reversal_21d", "window": 21},
]

# RESIDUAL MOMENTUM
RESIDUAL_MOMENTUM_CONFIGS = [
    {"variant": "resmom_6m_1m", "window": 126, "skip": 21},
    {"variant": "resmom_9m_1m", "window": 189, "skip": 21},
    {"variant": "resmom_12m_1m", "window": 252, "skip": 21},
]

# VOLATILITY-SCALED MOMENTUM
VOLATILITY_SCALED_MOMENTUM_CONFIGS = [
    {
        "variant": "vsmom_6m_1m_vol60",
        "window": 126,
        "skip": 21,
        "volatility_window": 60,
    },
    {
        "variant": "vsmom_9m_1m_vol60",
        "window": 189,
        "skip": 21,
        "volatility_window": 60,
    },
    {
        "variant": "vsmom_12m_1m_vol60",
        "window": 252,
        "skip": 21,
        "volatility_window": 60,
    },
]

# HIGH PROXIMITY
HIGH_PROXIMITY_CONFIGS = [
    {"variant": "high_6m", "window": 126},
    {"variant": "high_12m", "window": 252},
    {"variant": "high_18m", "window": 378},
]

# TREND SLOPE
TREND_SLOPE_CONFIGS = [
    {"variant": "slope_50d", "window": 50},
    {"variant": "slope_100d", "window": 100},
    {"variant": "slope_200d", "window": 200},
    {"variant": "slope_250d", "window": 250},
]

# RISK-ADJUSTED TREND
RISK_ADJUSTED_TREND_CONFIGS = [
    {"variant": "risk_trend_50d", "window": 50},
    {"variant": "risk_trend_100d", "window": 100},
    {"variant": "risk_trend_200d", "window": 200},
    {"variant": "risk_trend_250d", "window": 250},
]

# LIQUIDITY CHANGE
LIQUIDITY_CHANGE_CONFIGS = [
    {"variant": "liq_5d_60d", "short_window": 5, "long_window": 60},
    {"variant": "liq_20d_126d", "short_window": 20, "long_window": 126},
    {"variant": "liq_20d_252d", "short_window": 20, "long_window": 252},
]

# PRICE-VOLUME CONFIRMATION
PRICE_VOLUME_CONFIRMATION_CONFIGS = [
    {
        "variant": "pvc_6m_1m_liq5_60",
        "window": 126,
        "skip": 21,
        "short_window": 5,
        "long_window": 60,
        "confirmation_strength": 0.25,
        "liquidity_clip": 2,
    },
    {
        "variant": "pvc_9m_1m_liq20_126",
        "window": 189,
        "skip": 21,
        "short_window": 20,
        "long_window": 126,
        "confirmation_strength": 0.25,
        "liquidity_clip": 2,
    },
    {
        "variant": "pvc_12m_1m_liq20_252",
        "window": 252,
        "skip": 21,
        "short_window": 20,
        "long_window": 252,
        "confirmation_strength": 0.25,
        "liquidity_clip": 2,
    },
]


# -------------------------
# COMPLETE FACTOR GRID
FACTOR_CONFIGS = {
    "momentum": MOMENTUM_CONFIGS,
    "low_volatility": LOW_VOLATILITY_CONFIGS,
    "trend": TREND_CONFIGS,
    "short_term_reversal": SHORT_TERM_REVERSAL_CONFIGS,
    "residual_momentum": RESIDUAL_MOMENTUM_CONFIGS,
    "volatility_scaled_momentum": VOLATILITY_SCALED_MOMENTUM_CONFIGS,
    "high_proximity": HIGH_PROXIMITY_CONFIGS,
    "trend_slope": TREND_SLOPE_CONFIGS,
    "risk_adjusted_trend": RISK_ADJUSTED_TREND_CONFIGS,
    "liquidity_change": LIQUIDITY_CHANGE_CONFIGS,
    "price_volume_confirmation": PRICE_VOLUME_CONFIRMATION_CONFIGS,
}


FACTOR_VARIANT_COUNT = sum(
    len(configurations)
    for configurations in FACTOR_CONFIGS.values()
)

if FACTOR_VARIANT_COUNT != 56:
    raise ValueError("Factor configuration must contain exactly 56 variants")
