import numpy as np
import pandas as pd


def winsorize(df, lower=0.01, upper=0.99):
    lower_q = df.quantile(lower, axis=1)
    upper_q = df.quantile(upper, axis=1)

    return df.clip(lower=lower_q, upper=upper_q, axis=0)


def zscore(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    z = df.sub(mean, axis=0).div(std, axis=0)

    return z