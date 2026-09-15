import numpy as np
import pandas as pd

from portfolio_evaluation import annualized_return, annualized_sharpe
from research_config import (
    MARKET_NEUTRAL_METHODS,
    MIN_OOS_OBSERVATION_RATIO,
    MIN_SHORT_TRAIN_OBSERVATIONS,
    MIN_TRAIN_OBSERVATIONS,
    PRIMARY_TRANSACTION_COST_BPS,
    RESEARCH_START_DATE,
    WALK_FORWARD_SCHEMES,
)


def oriented_net_returns(gross, turnover, orientation):
    cost = turnover * PRIMARY_TRANSACTION_COST_BPS / 10_000
    return (1 + orientation * gross) * (1 - cost) - 1


def period_metrics(returns, risk_free, portfolio_type):
    returns = returns.dropna()
    aligned_rate = risk_free.reindex(returns.index)
    excess = (
        returns - aligned_rate
        if portfolio_type == "long_only"
        else returns
    )
    return {
        "observations": len(returns),
        "annualized_return": annualized_return(returns),
        "sharpe": annualized_sharpe(excess),
        "mean_daily_return": returns.mean() if len(returns) else np.nan,
    }


def walk_forward_periods(index, scheme_name, scheme):
    research_start = max(pd.Timestamp(RESEARCH_START_DATE), index.min())
    oos_start = research_start + pd.DateOffset(
        months=int(scheme["train_months"])
    )
    final_date = index.max()

    while oos_start <= final_date:
        train_start = oos_start - pd.DateOffset(
            months=int(scheme["train_months"])
        )
        train_end = oos_start - pd.Timedelta(days=1)
        oos_end = (
            oos_start
            + pd.DateOffset(months=int(scheme["oos_months"]))
            - pd.Timedelta(days=1)
        )
        expected_oos = len(index[(index >= oos_start) & (index <= oos_end)])
        yield {
            "walk_forward_scheme": scheme_name,
            "period_id": f"{oos_start:%Y-%m}_{oos_end:%Y-%m}",
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "calendar_complete": oos_end <= final_date,
            "expected_oos_observations": expected_oos,
        }
        oos_start += pd.DateOffset(months=int(scheme["step_months"]))


def run_walk_forward(matrices, metadata, risk_free):
    gross = matrices["gross_returns"]
    turnover = matrices["turnover"]
    metadata_lookup = metadata.set_index("path_key")
    if not metadata_lookup.index.is_unique:
        raise ValueError("Duplicated portfolio path metadata")
    period_rows = []
    stitched_columns = {}

    for scheme_name, scheme in WALK_FORWARD_SCHEMES.items():
        scheme_paths = pd.DataFrame(
            np.nan,
            index=gross.index,
            columns=gross.columns,
        )
        periods = list(walk_forward_periods(gross.index, scheme_name, scheme))
        minimum_train = (
            MIN_SHORT_TRAIN_OBSERVATIONS
            if scheme_name == "short"
            else MIN_TRAIN_OBSERVATIONS
        )

        for period in periods:
            train_start = period["train_start"]
            train_end = period["train_end"]
            oos_start = period["oos_start"]
            oos_end = period["oos_end"]

            for path in gross.columns:
                identity = metadata_lookup.loc[path]
                train_gross = gross.loc[train_start:train_end, path]
                train_turnover = turnover.loc[train_start:train_end, path]
                if len(train_gross) < minimum_train:
                    continue

                if identity["method"] in MARKET_NEUTRAL_METHODS:
                    positive = oriented_net_returns(
                        train_gross,
                        train_turnover,
                        1,
                    )
                    negative = oriented_net_returns(
                        train_gross,
                        train_turnover,
                        -1,
                    )
                    orientation = (
                        1 if positive.mean() >= negative.mean() else -1
                    )
                else:
                    orientation = 1

                train_net = oriented_net_returns(
                    train_gross,
                    train_turnover,
                    orientation,
                )
                oos_gross = gross.loc[oos_start:oos_end, path]
                oos_turnover = turnover.loc[oos_start:oos_end, path]
                oos_net = oriented_net_returns(
                    oos_gross,
                    oos_turnover,
                    orientation,
                )
                train_metrics = period_metrics(
                    train_net,
                    risk_free.loc[train_net.index],
                    identity["portfolio_type"],
                )
                oos_metrics = period_metrics(
                    oos_net,
                    risk_free.loc[oos_net.index],
                    identity["portfolio_type"],
                )
                expected = period["expected_oos_observations"]
                complete_oos_period = bool(
                    period["calendar_complete"]
                    and expected > 0
                    and oos_metrics["observations"] / expected
                    >= MIN_OOS_OBSERVATION_RATIO
                )
                if complete_oos_period:
                    scheme_paths.loc[oos_net.index, path] = oos_net
                period_rows.append(
                    {
                        "walk_key": f"{scheme_name}|{path}",
                        "path_key": path,
                        **identity.to_dict(),
                        **period,
                        "orientation_selected_in_training": orientation,
                        "complete_oos_period": complete_oos_period,
                        **{
                            f"train_{name}": value
                            for name, value in train_metrics.items()
                        },
                        **{
                            f"oos_{name}": value
                            for name, value in oos_metrics.items()
                        },
                    }
                )

        for path in scheme_paths.columns:
            stitched_columns[f"{scheme_name}|{path}"] = scheme_paths[path]

    periods = pd.DataFrame(period_rows)
    stitched = pd.DataFrame(stitched_columns, index=gross.index)
    summary_rows = []
    if not periods.empty:
        complete = periods.loc[periods["complete_oos_period"]].copy()
        for walk_key, group in complete.groupby("walk_key"):
            path = group.iloc[0]["path_key"]
            identity = metadata_lookup.loc[path].to_dict()
            orientations = group.sort_values("oos_start")[
                "orientation_selected_in_training"
            ]
            path_returns = stitched[walk_key].dropna()
            excess = (
                path_returns - risk_free.reindex(path_returns.index)
                if identity["portfolio_type"] == "long_only"
                else path_returns
            )
            summary_rows.append(
                {
                    "walk_key": walk_key,
                    "path_key": path,
                    **identity,
                    "walk_forward_scheme": group.iloc[0][
                        "walk_forward_scheme"
                    ],
                    "complete_oos_periods": len(group),
                    "positive_oos_period_rate": float(
                        (group["oos_annualized_return"] > 0).mean()
                    ),
                    "median_oos_annualized_return": group[
                        "oos_annualized_return"
                    ].median(),
                    "worst_oos_annualized_return": group[
                        "oos_annualized_return"
                    ].min(),
                    "best_oos_annualized_return": group[
                        "oos_annualized_return"
                    ].max(),
                    "median_oos_sharpe": group["oos_sharpe"].median(),
                    "stitched_oos_annualized_return": annualized_return(
                        path_returns
                    ),
                    "stitched_oos_sharpe": annualized_sharpe(excess),
                    "orientation_positive_rate": float(
                        (orientations > 0).mean()
                    ),
                    "orientation_changes": int(
                        orientations.diff().fillna(0).ne(0).sum()
                    ),
                    "post_selection_oos_warning": True,
                }
            )
    return periods, pd.DataFrame(summary_rows), stitched
