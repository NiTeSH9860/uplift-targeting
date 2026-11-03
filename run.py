import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data import make_synthetic, load_criteo, feature_cols
from src.checks import standardized_mean_diff, ate
from src.evaluation import (qini_curve, auuc, uplift_at_k, decile_table,
                            bootstrap_ci, paired_bootstrap_diff)
from src.models.learners import (NaiveClassifier, SLearner, TLearner, XLearner,
                                 PARAMS, SPARSE_PARAMS)

os.makedirs("results", exist_ok=True)

# 1. Data
parser = argparse.ArgumentParser()
parser.add_argument("--criteo", type=int, default=0,
                    help="number of Criteo rows to sample; 0 = synthetic data")
parser.add_argument("--target", default="conversion", choices=["conversion", "visit"],
                    help="Criteo outcome to model")
parser.add_argument("--params", default="default", choices=["default", "sparse"],
                    help="LightGBM settings; 'sparse' is for rare outcomes")
args = parser.parse_args()

tag = f"criteo_{args.target}" if args.criteo else "synthetic"
if args.params == "sparse":
    tag += "_sparse"
params = SPARSE_PARAMS if args.params == "sparse" else PARAMS

df = (load_criteo(sample=args.criteo, target=args.target)
      if args.criteo else make_synthetic(n=100_000))
feats = feature_cols(df)
print(f"Dataset: {tag}  |  rows: {len(df):,}  |  outcome rate: {df['y'].mean():.4f}\n")

# 2. Sanity checks
print("== Randomization check (|SMD| > 0.1 is a red flag) ==")
print(standardized_mean_diff(df, feats).round(4), "\n")

print("== Average treatment effect ==")
result = ate(df)
print(f"ATE = {result['ate']:.4f}  (95% CI {result['ci_low']:.4f} to {result['ci_high']:.4f})\n")

# 3. Train / test split
train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df["treatment"])
X_tr, t_tr, y_tr = train[feats].values, train["treatment"].values, train["y"].values
X_te, t_te, y_te = test[feats].values, test["treatment"].values, test["y"].values

# 4. Fit models and evaluate (every model uses the same parameter set)
models = {
    "Naive P(buy)": NaiveClassifier(params),
    "S-learner": SLearner(params),
    "T-learner": TLearner(params),
    "X-learner": XLearner(params),
}

rows, scores = [], {}
for name, model in models.items():
    model.fit(X_tr, t_tr, y_tr)
    s = model.predict_uplift(X_te)
    scores[name] = s

    a = bootstrap_ci(auuc, s, t_te, y_te, n_boot=100)
    k = bootstrap_ci(uplift_at_k, s, t_te, y_te, n_boot=100, k=0.3)
    rows.append({
        "model": name,
        "AUUC": round(a["mean"], 1),
        "AUUC_low": round(a["low"], 1),
        "AUUC_high": round(a["high"], 1),
        "uplift@30%": round(k["mean"], 4),
        "corr_with_truth": (round(np.corrcoef(s, test["true_uplift"].values)[0, 1], 3)
                            if "true_uplift" in test else None),
    })

results = pd.DataFrame(rows).set_index("model")
print("== Test-set results ==")
print(results.to_string(), "\n")
results.to_csv(f"results/results_{tag}.csv")

# 5. Paired comparison against the naive baseline (same resampled customers)
baseline = "Naive P(buy)"
paired_rows = []
for name in scores:
    if name == baseline:
        continue
    d = paired_bootstrap_diff(auuc, scores[name], scores[baseline], t_te, y_te, n_boot=100)
    paired_rows.append({
        "model": name,
        "AUUC_gain_vs_naive": round(d["mean_diff"], 1),
        "low": round(d["low"], 1),
        "high": round(d["high"], 1),
        "prob_better": round(d["prob_a_better"], 2),
    })
paired = pd.DataFrame(paired_rows).set_index("model")
print("== Paired AUUC difference vs naive baseline (95% CI) ==")
print(paired.to_string(), "\n")
paired.to_csv(f"results/paired_{tag}.csv")

# 6. Qini curve plot
plt.figure(figsize=(7, 5))
for name, s in scores.items():
    x, q = qini_curve(s, t_te, y_te)
    plt.plot(x, q, label=name)
plt.plot([0, 1], [0, q[-1]], "k--", label="Random targeting")
plt.xlabel("Fraction of customers targeted")
plt.ylabel("Incremental conversions")
plt.title(f"Qini curves ({tag})")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/qini_curves_{tag}.png", dpi=150)

# 7. Decile table for the best model
best = results["AUUC"].idxmax()
print(f"== Decile table: {best} ==")
print(decile_table(scores[best], t_te, y_te).round(4))