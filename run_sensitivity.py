import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import make_synthetic, load_criteo, feature_cols
from src.policy import profit_curve, best_cutoff, profit_at_fraction
from src.models.learners import (NaiveClassifier, SLearner, TLearner, XLearner,
                                 PARAMS, SPARSE_PARAMS)

os.makedirs("results", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--criteo", type=int, default=0,
                    help="number of Criteo rows to sample; 0 = synthetic data")
parser.add_argument("--target", default="visit", choices=["conversion", "visit"])
parser.add_argument("--params", default="default", choices=["default", "sparse"])
parser.add_argument("--value", type=float, default=None)
parser.add_argument("--costs", type=float, nargs="+", default=None)
args = parser.parse_args()

value = args.value if args.value is not None else (1.0 if args.criteo else 10.0)
costs = args.costs or ([0.002, 0.005, 0.01, 0.02, 0.03] if args.criteo
                       else [0.1, 0.3, 0.5, 0.7])

tag = f"criteo_{args.target}" if args.criteo else "synthetic"
if args.params == "sparse":
    tag += "_sparse"
params = SPARSE_PARAMS if args.params == "sparse" else PARAMS

df = (load_criteo(sample=args.criteo, target=args.target)
      if args.criteo else make_synthetic(n=200_000))
feats = feature_cols(df)
print(f"Dataset: {tag}  |  rows: {len(df):,}  |  value={value}  costs={costs}\n")

# Train 50% / validation 20% / test 30%
train, rest = train_test_split(df, test_size=0.5, random_state=0, stratify=df["treatment"])
valid, test = train_test_split(rest, test_size=0.6, random_state=0, stratify=rest["treatment"])

def parts(d):
    return d[feats].values, d["treatment"].values, d["y"].values

X_tr, t_tr, y_tr = parts(train)
X_va, t_va, y_va = parts(valid)
X_te, t_te, y_te = parts(test)

models = {
    "Naive P(buy)": NaiveClassifier(params),
    "S-learner": SLearner(params),
    "T-learner": TLearner(params),
    "X-learner": XLearner(params),
}

# Fit once: scores do not depend on the cost
scores_va, scores_te = {}, {}
for name, model in models.items():
    model.fit(X_tr, t_tr, y_tr)
    scores_va[name] = model.predict_uplift(X_va)
    scores_te[name] = model.predict_uplift(X_te)

everyone = np.zeros(len(t_te))  # constant score: contacting the top 100% = everyone


def paired_profit_diff(s_a, frac_a, s_b, frac_b, cost, n_boot=100, seed=0):
    """95% CI for profit(A) - profit(B), scoring both on the same resampled customers."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(t_te), len(t_te))
        a = profit_at_fraction(s_a[idx], t_te[idx], y_te[idx], frac_a, value, cost)
        b = profit_at_fraction(s_b[idx], t_te[idx], y_te[idx], frac_b, value, cost)
        diffs.append(a - b)
    diffs = np.array(diffs)
    return (round(np.nanmean(diffs), 1), round(np.nanpercentile(diffs, 2.5), 1),
            round(np.nanpercentile(diffs, 97.5), 1))


profit_rows, frac_rows, diff_rows = [], [], []
for cost in costs:
    prow = {"cost": cost, "Contact everyone": round(
        profit_at_fraction(everyone, t_te, y_te, 1.0, value, cost), 1)}
    frow = {"cost": cost}
    fracs = {}

    for name in models:
        # Cutoff chosen on validation only; profit reported on test
        frac = float(best_cutoff(profit_curve(scores_va[name], t_va, y_va, value, cost))["frac"])
        fracs[name] = frac
        frow[name] = round(frac, 3)
        prow[name] = round(profit_at_fraction(scores_te[name], t_te, y_te, frac, value, cost), 1)

    profit_rows.append(prow)
    frac_rows.append(frow)

    for label, s_b, f_b in [("S minus Naive", scores_te["Naive P(buy)"], fracs["Naive P(buy)"]),
                            ("S minus Everyone", everyone, 1.0)]:
        m, lo, hi = paired_profit_diff(scores_te["S-learner"], fracs["S-learner"],
                                       s_b, f_b, cost)
        diff_rows.append({"cost": cost, "comparison": label,
                          "profit_diff": m, "ci_low": lo, "ci_high": hi})

profit_df = pd.DataFrame(profit_rows).set_index("cost")
frac_df = pd.DataFrame(frac_rows).set_index("cost")
diff_df = pd.DataFrame(diff_rows).set_index(["cost", "comparison"])

print("== Test profit per 1,000 customers, by contact cost ==")
print(profit_df.to_string(), "\n")
print("== Fraction of customers contacted ==")
print(frac_df.to_string(), "\n")
print("== Paired profit difference per 1,000 (95% CI) ==")
print(diff_df.to_string())

profit_df.to_csv(f"results/sensitivity_profit_{tag}.csv")
diff_df.to_csv(f"results/sensitivity_paired_{tag}.csv")