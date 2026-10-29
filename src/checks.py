import numpy as np
import pandas as pd


def standardized_mean_diff(df, cols, treat="treatment"):
    """Compare each feature between treated and control groups."""
    t = df[df[treat] == 1]
    c = df[df[treat] == 0]

    out = {}
    for col in cols:
        pooled_sd = np.sqrt((t[col].var() + c[col].var()) / 2)
        out[col] = (t[col].mean() - c[col].mean()) / pooled_sd if pooled_sd > 0 else 0.0

    result = pd.Series(out, name="SMD").to_frame()
    result["flag"] = result["SMD"].abs() > 0.1  # True = imbalance
    return result


def ate(df, treat="treatment", y="y", z=1.96):
    """Average treatment effect: treated rate minus control rate, with 95% CI."""
    t = df.loc[df[treat] == 1, y]
    c = df.loc[df[treat] == 0, y]

    estimate = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))

    return {
        "ate": estimate,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "rate_treat": t.mean(),
        "rate_control": c.mean(),
        "n_treat": len(t),
        "n_control": len(c),
    }