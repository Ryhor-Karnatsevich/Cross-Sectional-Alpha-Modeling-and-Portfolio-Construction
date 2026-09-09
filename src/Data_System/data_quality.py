import pandas as pd


def build_point_in_time_quality_mask(
    prices,
    membership,
    max_abs_daily_return,
    max_extreme_daily_returns,
):
    """Build a data-quality mask using only information known by each date."""
    prices, membership = prices.align(membership, join="left")
    membership = membership.fillna(False).astype(bool)

    observed_positive_price = prices.notna() & prices.gt(0)
    raw_returns = prices.pct_change(fill_method=None)
    extreme_returns = raw_returns.abs().gt(max_abs_daily_return) & membership

    # A ticker remains usable through the allowed number of extreme observations.
    # Once the limit is exceeded, it is quarantined from that date forward only.
    extreme_count_to_date = extreme_returns.cumsum()
    quarantined = extreme_count_to_date.gt(max_extreme_daily_returns)

    quality = observed_positive_price & ~quarantined
    return quality.astype(bool), extreme_returns.astype(bool), quarantined.astype(bool)


def first_true_date(mask):
    """Return the first True date for every column without looking backward."""
    result = {}

    for ticker in mask.columns:
        dates = mask.index[mask[ticker]]
        result[ticker] = dates[0] if len(dates) else pd.NaT

    return pd.Series(result)
