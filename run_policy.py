import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import make_synthetic, load_criteo, feature_cols
from src.evaluation import bootstrap_ci
from src.policy import profit_curve, best_cutoff, profit_at_fraction
from src.models.learners import (NaiveClassifier, SLearner, TLearner, XLearner,
                                 PARAMS, SPARSE_PARAMS)

os.makedirs("results", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--criteo", type=int, default=0,
                    help="number of Criteo rows to sample; 0 = synthetic data")
parser.add_argument("--target", default="visit", choices=["conversion", "visit"])
parser.add_argument("--params", default="default", choices=["default", "sparse"])
parser.add_argument("--value", type=float, default=None,
                    help="profit from one incremental conversion/visit (assumption)")
parser.add_argument("--cost", type=float, default=None,
                    help="cost of contacting one customer (assumption)")
args = parser.parse_args()

# Assumptions (illustrative defaults: replace with your own and state them in the README)
value = args.value if args.value is not None else (1.0 if args.criteo else 10.0)
cost = args.cost if args.cost is not None else (0.01 if args.criteo else 0.5)

tag = f"criteo_{args.target}" if args.criteo else "synthetic"
if args.params == "sparse":
    tag += "_sparse"
params = SPARSE_PARAMS if args.params == "sparse" else PARAMS

df = (load_criteo(sample=args.criteo, target=args.target)
      if args.criteo else make_synthetic(n=200_000))
feats = feature_cols(df)
print(f"Dataset: {tag}  |  rows: {len(df):,}  |  value={value}  cost={cost}\n")

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

def report(name, frac, s_test):
    point = profit_at_fraction(s_test, t_te, y_te, frac, value, cost)
    ci = bootstrap_ci(profit_at_fraction, s_test, t_te, y_te, n_boot=200,
                      frac=frac, value=value, cost=cost)
    return {"policy": name, "frac_contacted": round(frac, 3),
            "profit_per_1000": round(point, 1),
            "ci_low": round(ci["low"], 1), "ci_high": round(ci["high"], 1)}

rows = [
    {"policy": "Contact nobody", "frac_contacted": 0.0,
     "profit_per_1000": 0.0, "ci_low": 0.0, "ci_high": 0.0},
    report("Contact everyone", 1.0, np.zeros(len(t_te))),
]

for name, model in models.items():
    model.fit(X_tr, t_tr, y_tr)
    # Choose the cutoff on the validation set only
    s_va = model.predict_uplift(X_va)
    frac = float(best_cutoff(profit_curve(s_va, t_va, y_va, value, cost))["frac"])
    # Report profit of that fixed cutoff on the held-out test set
    rows.append(report(name, frac, model.predict_uplift(X_te)))

results = pd.DataFrame(rows).set_index("policy")
print("== Test-set profit per 1,000 customers (95% CI) ==")
print(results.to_string())
results.to_csv(f"results/policy_{tag}.csv")