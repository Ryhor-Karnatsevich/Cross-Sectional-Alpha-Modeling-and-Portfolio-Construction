import pandas as pd


def build_confirmed_event_mask(prices, confirmed_real_return_events):
    confirmed = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    for ticker, date in confirmed_real_return_events or ():
        timestamp = pd.Timestamp(date)
        if ticker in confirmed.columns and timestamp in confirmed.index:
            confirmed.loc[timestamp, ticker] = True

    return confirmed


def build_data_quality_mask(
    prices,
    suspicious_abs_daily_return,
    max_abs_daily_return,
    round_trip_return_tolerance,
    confirmed_real_return_events=None,
):
    """Separate real large moves from unconfirmed price-data anomalies."""
    prices = prices.sort_index()

    observed_positive_price = prices.notna() & prices.gt(0)
    raw_returns = prices.pct_change(fill_method=None)
    suspicious_returns = raw_returns.abs().ge(suspicious_abs_daily_return)
    extreme_returns = raw_returns.abs().ge(max_abs_daily_return)

    next_returns = raw_returns.shift(-1)
    round_trip_returns = prices.shift(-1).div(prices.shift(1)) - 1
    spike_reversals = (
        suspicious_returns
        & next_returns.abs().ge(suspicious_abs_daily_return)
        & round_trip_returns.abs().le(round_trip_return_tolerance)
    )

    confirmed = build_confirmed_event_mask(prices, confirmed_real_return_events)
    anomaly_triggers = (extreme_returns | spike_reversals) & ~confirmed

    # An unconfirmed anomaly is never converted into an artificial return.
    # The ticker is excluded from the first suspect price onward. A one-day
    # reversal is retrospective data cleaning, not a trading feature.
    quarantined = anomaly_triggers.cummax()

    quality = observed_positive_price & ~quarantined
    return (
        quality.astype(bool),
        suspicious_returns.astype(bool),
        anomaly_triggers.astype(bool),
        quarantined.astype(bool),
    )


def first_true_date(mask):
    """Return the first True date for every column without looking backward."""
    result = {}

    for ticker in mask.columns:
        dates = mask.index[mask[ticker]]
        result[ticker] = dates[0] if len(dates) else pd.NaT

    return pd.Series(result)
