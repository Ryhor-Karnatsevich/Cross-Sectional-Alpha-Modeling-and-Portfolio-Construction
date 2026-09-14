import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from selection_config import (
    ACTIVE_FACTOR_KEYS,
    EFFECT_NAMES,
    FORWARD_HORIZONS,
    MIDDLE_QUANTILES,
    MIN_ASSETS,
    QUANTILE_COUNT,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    SIGNAL_LAG,
)
from selection_storage import (
    load_factor_matrix,
    load_factor_metadata,
    load_factor_quantile_results,
    load_membership,
)


# -------------------------
# ANALYSIS SCOPE
def select_factor_metadata(metadata):
    if ACTIVE_FACTOR_KEYS is None:
        return metadata.copy()

    requested = list(ACTIVE_FACTOR_KEYS)
    selected = metadata[metadata["key"].isin(requested)].copy()
    missing = sorted(set(requested).difference(selected["key"]))

    if missing:
        raise ValueError(
            "Configured factors do not exist: " + ", ".join(missing)
        )

    return selected.set_index("key").loc[requested].reset_index()


def required_quantile_columns():
    columns = [
        "date",
        "hypothesis_key",
        "factor_key",
        "family",
        "variant",
        "horizon_days",
        "parameters",
        "research_eligible",
        "membership_count",
        "signal_asset_count",
        "return_asset_count",
        "spearman_ic",
    ]

    for quantile in range(1, QUANTILE_COUNT + 1):
        columns.extend(
            [
                f"q{quantile}_count",
                f"q{quantile}_factor_score_mean",
                f"q{quantile}_return_mean",
                f"q{quantile}_return_median",
            ]
        )

    return columns


# -------------------------
# STATISTICAL HELPERS
def hac_tstat(values, max_lag):
    values = pd.Series(values).dropna().astype(float)
    observation_count = len(values)

    if observation_count < 2:
        return np.nan

    demeaned = values.to_numpy() - values.mean()
    max_lag = min(max(0, int(max_lag)), observation_count - 1)
    long_run_variance = np.dot(demeaned, demeaned) / observation_count

    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        covariance = (
            np.dot(demeaned[lag:], demeaned[:-lag]) / observation_count
        )
        long_run_variance += 2 * weight * covariance

    if long_run_variance <= 0:
        return np.nan

    standard_error = np.sqrt(long_run_variance / observation_count)
    return float(values.mean() / standard_error)


def direction_rate(values, direction):
    values = pd.Series(values).dropna()

    if values.empty or direction == 0:
        return np.nan

    return float((np.sign(values) == direction).mean())


def period_means(dates, values, frequency):
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates)),
            "value": np.asarray(values, dtype=float),
        }
    ).dropna(subset=["value"])

    if data.empty:
        return pd.Series(dtype=float)

    data["period"] = data["date"].dt.to_period(frequency)
    return data.groupby("period", sort=True)["value"].mean()


def summarize_effect(dates, values, horizon):
    values = pd.Series(
        np.asarray(values, dtype=float),
        index=pd.DatetimeIndex(np.asarray(dates)),
    )
    valid = values.dropna()
    observation_count = len(valid)
    research_dates = len(values)
    mean = valid.mean() if observation_count else np.nan
    direction = int(np.sign(mean)) if pd.notna(mean) and mean != 0 else 0
    tstat = hac_tstat(valid, int(horizon) - 1)
    monthly = period_means(valid.index, valid.values, "M")
    annual = period_means(valid.index, valid.values, "Y")

    summary = {
        "observations": observation_count,
        "valid_date_ratio": (
            observation_count / research_dates if research_dates else np.nan
        ),
        "mean": mean,
        "median": valid.median() if observation_count else np.nan,
        "std": valid.std() if observation_count > 1 else np.nan,
        "hac_tstat": tstat,
        "raw_p_value": (
            float(2 * norm.sf(abs(tstat))) if pd.notna(tstat) else np.nan
        ),
        "direction": direction,
        "daily_direction_rate": direction_rate(valid, direction),
        "monthly_direction_rate": direction_rate(monthly, direction),
        "annual_direction_rate": direction_rate(annual, direction),
        "monthly_periods": len(monthly),
        "annual_periods": len(annual),
        "worst_annual_mean": annual.min() if len(annual) else np.nan,
        "best_annual_mean": annual.max() if len(annual) else np.nan,
    }
    return summary, monthly, annual


# -------------------------
# DAILY ECONOMIC EFFECTS
def daily_effects(frame):
    middle_columns = [
        f"q{quantile}_return_mean" for quantile in MIDDLE_QUANTILES
    ]
    middle = frame[middle_columns].mean(axis=1)
    q1 = frame["q1_return_mean"]
    q10 = frame[f"q{QUANTILE_COUNT}_return_mean"]

    return {
        "spearman_ic": frame["spearman_ic"],
        "q10_minus_q1": q10 - q1,
        "q10_minus_middle": q10 - middle,
        "middle_minus_q1": middle - q1,
        "edges_minus_middle": (q1 + q10) / 2 - middle,
    }


def quantile_curve(frame):
    rows = []

    for quantile in range(1, QUANTILE_COUNT + 1):
        rows.append(
            {
                "quantile": quantile,
                "average_count": frame[f"q{quantile}_count"].mean(),
                "average_factor_score": frame[
                    f"q{quantile}_factor_score_mean"
                ].mean(),
                "average_return": frame[
                    f"q{quantile}_return_mean"
                ].mean(),
                "median_daily_return": frame[
                    f"q{quantile}_return_median"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def curve_shape_statistics(curve):
    returns = curve["average_return"].to_numpy(dtype=float)

    if not np.isfinite(returns).all() or np.unique(returns).size < 2:
        return np.nan, np.nan

    rho = float(
        spearmanr(
            np.arange(1, QUANTILE_COUNT + 1),
            returns,
        ).statistic
    )
    direction = np.sign(rho)
    steps = np.diff(returns)
    step_ratio = (
        float((np.sign(steps) == direction).mean())
        if direction != 0
        else np.nan
    )
    return rho, step_ratio


# -------------------------
# FACTOR RANK STABILITY
def rank_autocorrelation_by_horizon(factor, membership):
    signal = factor.shift(SIGNAL_LAG).where(membership)
    ranks = signal.rank(axis=1, method="average", pct=True)
    research_end = RESEARCH_END_DATE or ranks.index.max()
    rows = {}

    for horizon in FORWARD_HORIZONS:
        previous = ranks.shift(int(horizon))
        valid = ranks.notna() & previous.notna()
        correlations = ranks.where(valid).corrwith(
            previous.where(valid),
            axis=1,
        )
        correlations = correlations.where(valid.sum(axis=1) >= MIN_ASSETS)
        correlations = correlations.loc[RESEARCH_START_DATE:research_end]
        valid_correlations = correlations.dropna()
        rows[int(horizon)] = {
            "rank_autocorrelation_observations": len(valid_correlations),
            "mean_rank_autocorrelation": valid_correlations.mean(),
            "median_rank_autocorrelation": valid_correlations.median(),
        }

    return rows


# -------------------------
# ONE FACTOR CONFIGURATION
def analyze_factor(factor_information, membership):
    factor_key = factor_information.key
    factor_results = load_factor_quantile_results(
        factor_key,
        required_quantile_columns(),
    )
    factor_results["date"] = pd.to_datetime(factor_results["date"])
    research = factor_results[factor_results["research_eligible"]].copy()
    factor = load_factor_matrix(
        factor_information.family,
        factor_information.variant,
        membership,
    )
    rank_stability = rank_autocorrelation_by_horizon(factor, membership)
    card_rows = []
    effect_rows = []
    curve_rows = []
    time_rows = []

    for horizon, frame in research.groupby("horizon_days", sort=True):
        horizon = int(horizon)
        frame = frame.sort_values("date")
        effects = daily_effects(frame)
        effect_summaries = {}

        for effect_name in EFFECT_NAMES:
            summary, monthly, annual = summarize_effect(
                frame["date"],
                effects[effect_name],
                horizon,
            )
            effect_summaries[effect_name] = summary
            effect_rows.append(
                {
                    "hypothesis_key": frame["hypothesis_key"].iloc[0],
                    "factor_key": factor_key,
                    "family": factor_information.family,
                    "variant": factor_information.variant,
                    "horizon_days": horizon,
                    "effect": effect_name,
                    **summary,
                }
            )

            for frequency, values in (
                ("monthly", monthly),
                ("annual", annual),
            ):
                for period, value in values.items():
                    time_rows.append(
                        {
                            "hypothesis_key": frame[
                                "hypothesis_key"
                            ].iloc[0],
                            "factor_key": factor_key,
                            "horizon_days": horizon,
                            "effect": effect_name,
                            "frequency": frequency,
                            "period": str(period),
                            "mean_effect": value,
                        }
                    )

        curve = quantile_curve(frame)
        curve_rho, step_ratio = curve_shape_statistics(curve)
        curve.insert(0, "horizon_days", horizon)
        curve.insert(0, "factor_key", factor_key)
        curve_rows.append(curve)
        ic_summary = effect_summaries["spearman_ic"]
        card = {
            "hypothesis_key": frame["hypothesis_key"].iloc[0],
            "factor_key": factor_key,
            "family": factor_information.family,
            "variant": factor_information.variant,
            "horizon_days": horizon,
            "parameters": factor_information.parameters,
            "research_dates": len(frame),
            "median_membership_count": frame["membership_count"].median(),
            "median_return_asset_count": frame[
                "return_asset_count"
            ].median(),
            "valid_ic_ratio": ic_summary["valid_date_ratio"],
            "quantile_curve_rho": curve_rho,
            "quantile_step_ratio": step_ratio,
            **rank_stability[horizon],
        }

        for effect_name, summary in effect_summaries.items():
            for metric in (
                "mean",
                "median",
                "hac_tstat",
                "raw_p_value",
                "daily_direction_rate",
                "monthly_direction_rate",
                "annual_direction_rate",
            ):
                card[f"{effect_name}_{metric}"] = summary[metric]

        card_rows.append(card)

    found_horizons = set(research["horizon_days"].unique().astype(int))
    if found_horizons != set(FORWARD_HORIZONS):
        raise ValueError(f"Incomplete horizons for {factor_key}")

    return (
        pd.DataFrame(card_rows),
        pd.DataFrame(effect_rows),
        pd.concat(curve_rows, ignore_index=True),
        pd.DataFrame(time_rows),
    )


# -------------------------
# COMPLETE HYPOTHESIS METRICS
def build_hypothesis_metrics():
    metadata = select_factor_metadata(load_factor_metadata())
    membership = load_membership()
    card_tables = []
    effect_tables = []
    curve_tables = []
    time_tables = []

    for number, factor_information in enumerate(
        metadata.itertuples(index=False),
        start=1,
    ):
        print(
            f"Analyze factor {number}/{len(metadata)}: "
            f"{factor_information.key}"
        )
        cards, effects, curves, times = analyze_factor(
            factor_information,
            membership,
        )
        card_tables.append(cards)
        effect_tables.append(effects)
        curve_tables.append(curves)
        time_tables.append(times)

    return (
        pd.concat(card_tables, ignore_index=True),
        pd.concat(effect_tables, ignore_index=True),
        pd.concat(curve_tables, ignore_index=True),
        pd.concat(time_tables, ignore_index=True),
        len(metadata),
    )
