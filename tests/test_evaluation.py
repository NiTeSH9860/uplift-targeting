import numpy as np

from src.data import make_synthetic, feature_cols
from src.checks import standardized_mean_diff, ate
from src.evaluation import auuc, uplift_at_k, qini_curve


def test_randomized_data_is_balanced():
    df = make_synthetic(n=30_000, seed=1)
    smd = standardized_mean_diff(df, feature_cols(df))
    assert not smd["flag"].any()


def test_ate_interval_contains_true_effect():
    df = make_synthetic(n=200_000, seed=2)
    result = ate(df)
    true_ate = df["true_uplift"].mean()
    assert result["ci_low"] <= true_ate <= result["ci_high"]


def test_oracle_beats_random_and_reversed():
    df = make_synthetic(n=100_000, seed=3)
    t, y = df["treatment"].values, df["y"].values

    oracle = auuc(df["true_uplift"].values, t, y)
    random_scores = np.random.default_rng(0).random(len(df))
    random_ = auuc(random_scores, t, y)
    reversed_ = auuc(-df["true_uplift"].values, t, y)

    assert oracle > 0
    assert oracle > random_
    assert reversed_ < 0


def test_oracle_top_20_percent_has_high_uplift():
    df = make_synthetic(n=100_000, seed=4)
    u = uplift_at_k(df["true_uplift"].values, df["treatment"].values, df["y"].values, k=0.2)
    assert u > 0.06  # persuadables have a true effect of 0.08


def test_qini_curve_endpoints():
    df = make_synthetic(n=10_000, seed=5)
    x, q = qini_curve(df["true_uplift"].values, df["treatment"].values, df["y"].values)
    assert x[0] == 0 and q[0] == 0 and x[-1] == 1