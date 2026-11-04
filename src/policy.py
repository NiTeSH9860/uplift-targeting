import numpy as np
import pandas as pd


def profit_curve(score, t, y, value, cost, n_points=200):
    """Profit from targeting the top-k customers, for a grid of k.

    value: profit from one incremental conversion
    cost:  cost of contacting one customer
    Starts at 1% of customers: below that the control group is too small
    for a stable uplift estimate.
    """
    order = np.argsort(-np.asarray(score), kind="stable")
    t = np.asarray(t)[order]
    y = np.asarray(y)[order]
    n = len(t)

    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    conv_t, conv_c = np.cumsum(y * t), np.cumsum(y * (1 - t))

    start = max(1, int(0.01 * n))
    m = np.unique(np.linspace(start, n, n_points).astype(int))
    i = m - 1

    uplift = conv_t[i] / np.maximum(cum_t[i], 1) - conv_c[i] / np.maximum(cum_c[i], 1)
    incremental = m * uplift
    profit = value * incremental - cost * m

    return pd.DataFrame({
        "frac": m / n,
        "n_targeted": m,
        "uplift": uplift,
        "incremental_conversions": incremental,
        "profit": profit,
        "profit_per_1000": 1000 * profit / n,   # per 1,000 customers in the population
    })

def best_cutoff(curve):
    """Row of the curve with the highest profit, or 'contact nobody' if all lose money."""
    best = curve.loc[curve["profit"].idxmax()]
    if best["profit"] <= 0:
        return pd.Series({"frac": 0.0, "n_targeted": 0, "uplift": np.nan,
                          "incremental_conversions": 0.0, "profit": 0.0,
                          "profit_per_1000": 0.0})
    return best

def profit_at_fraction(score, t, y, frac, value, cost):
    """Profit per 1,000 customers from contacting the top `frac` by score."""
    score, t, y = np.asarray(score), np.asarray(t), np.asarray(y)
    n = len(t)
    m = int(frac * n)
    if m == 0:
        return 0.0

    top = np.argsort(-score, kind="stable")[:m]
    t_top, y_top = t[top], y[top]
    if t_top.sum() == 0 or (1 - t_top).sum() == 0:
        return np.nan

    uplift = y_top[t_top == 1].mean() - y_top[t_top == 0].mean()
    return 1000 * m * (value * uplift - cost) / n