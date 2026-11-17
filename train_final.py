import argparse
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.data import load_criteo, feature_cols
from src.models.learners import SLearner, PARAMS

parser = argparse.ArgumentParser()
parser.add_argument("--criteo", type=int, default=1_000_000,
                    help="number of Criteo rows to sample")
parser.add_argument("--target", default="visit", choices=["conversion", "visit"])
parser.add_argument("--out", default="models/s_learner_criteo_visit.joblib")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

df = load_criteo(sample=args.criteo, target=args.target)
feats = feature_cols(df)
print(f"Loaded {len(df):,} rows, {len(feats)} features, target={args.target}")

# Same split as run.py so the model and the reported results match
train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df["treatment"])

model = SLearner(PARAMS).fit(train[feats].values,
                             train["treatment"].values,
                             train["y"].values)

# Distribution of predicted uplift on held-out customers, used by the API
# to turn a raw score into a percentile ("top 8% of customers")
test_scores = model.predict_uplift(test[feats].values)
quantiles = np.quantile(test_scores, np.linspace(0, 1, 101))

artifact = {
    "model": model,
    "features": feats,
    "score_quantiles": quantiles,
    "target": args.target,
    "n_train": len(train),
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
joblib.dump(artifact, args.out)

print(f"Saved model to {args.out}")
print(f"Test-set score range: {quantiles[0]:.4f} to {quantiles[-1]:.4f}  "
      f"(median {quantiles[50]:.4f}, 90th percentile {quantiles[90]:.4f})")
      