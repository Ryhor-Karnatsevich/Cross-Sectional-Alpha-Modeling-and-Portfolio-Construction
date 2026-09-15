import numpy as np
import pandas as pd

from portfolio_evaluation import annualized_sharpe


REGIME_COLUMNS = (
    "risk_free_state",
    "volatility_state",
    "dispersion_state",
    "correlation_state",
)


def run_regime_analysis(walk_forward_paths, regimes, walk_forward_summary):
    metadata_lookup = walk_forward_summary.set_index("walk_key")
    if not metadata_lookup.index.is_unique:
        raise ValueError("Duplicated walk-forward path metadata")
    rows = []

    for walk_key in walk_forward_paths.columns:
        if walk_key not in metadata_lookup.index:
            continue
        returns = walk_forward_paths[walk_key]
        identity = metadata_lookup.loc[walk_key].to_dict()
        for regime_column in REGIME_COLUMNS:
            states = regimes[regime_column].reindex(returns.index)
            for state in states.dropna().unique():
                if state == "unavailable":
                    continue
                values = returns.loc[states.eq(state)].dropna()
                rows.append(
                    {
                        "walk_key": walk_key,
                        **identity,
                        "regime_variable": regime_column,
                        "regime_state": state,
                        "observations": len(values),
                        "mean_daily_return": (
                            values.mean() if len(values) else np.nan
                        ),
                        "annualized_return_approx": (
                            values.mean() * 252 if len(values) else np.nan
                        ),
                        "sharpe": annualized_sharpe(values),
                        "positive_day_rate": (
                            (values > 0).mean() if len(values) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)
