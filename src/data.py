import numpy as np
import pandas as pd


def make_synthetic(n=50_000, p_treat=0.5, seed=0):
    """Fake randomized marketing experiment with a KNOWN true effect."""
    rng = np.random.default_rng(seed)

    # 10 random customer features
    X = rng.normal(size=(n, 10))

    # Baseline chance of buying WITHOUT the campaign
    base_logit = -2.5 + 0.6 * X[:, 2] + 0.4 * X[:, 3]
    base = 1 / (1 + np.exp(-base_logit))

    # True effect of the campaign on each customer
    #   x0 > 0.5  -> persuadables:   +8 percentage points
    #   x1 < -1.0 -> sleeping dogs:  -4 percentage points
    tau = np.where(X[:, 0] > 0.5, 0.08, 0.0) + np.where(X[:, 1] < -1.0, -0.04, 0.0)

    # Random assignment: a coin flip decides who gets the campaign
    t = rng.binomial(1, p_treat, size=n)

    # Did they buy? Depends on baseline + effect (only if treated)
    y = rng.binomial(1, np.clip(base + t * tau, 0, 1))

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["treatment"] = t
    df["y"] = y
    df["true_uplift"] = tau  # never available in real life
    return df


def feature_cols(df):
    drop = {"treatment", "y", "true_uplift"}
    return [c for c in df.columns if c not in drop]