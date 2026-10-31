import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

# Shared settings: small, regularized trees so we don't overfit noisy conversions
PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=0,
)


class NaiveClassifier:
    """Baseline: rank customers by P(buy). Ignores the campaign entirely."""

    def fit(self, X, t, y):
        self.model = LGBMClassifier(**PARAMS).fit(X, y)
        return self

    def predict_uplift(self, X):
        return self.model.predict_proba(X)[:, 1]


class SLearner:
    """One model with treatment as a feature."""

    def fit(self, X, t, y):
        self.model = LGBMClassifier(**PARAMS).fit(np.column_stack([X, t]), y)
        return self

    def predict_uplift(self, X):
        n = len(X)
        p_treated = self.model.predict_proba(np.column_stack([X, np.ones(n)]))[:, 1]
        p_control = self.model.predict_proba(np.column_stack([X, np.zeros(n)]))[:, 1]
        return p_treated - p_control


class TLearner:
    """Two models: one trained on treated customers, one on control."""

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.model_t = LGBMClassifier(**PARAMS).fit(X[t == 1], y[t == 1])
        self.model_c = LGBMClassifier(**PARAMS).fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]


class XLearner:
    """Uses imputed individual effects so every customer helps train the final model."""

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.p = t.mean()

        mu_t = LGBMClassifier(**PARAMS).fit(X[t == 1], y[t == 1])
        mu_c = LGBMClassifier(**PARAMS).fit(X[t == 0], y[t == 0])

        effect_treated = y[t == 1] - mu_c.predict_proba(X[t == 1])[:, 1]
        effect_control = mu_t.predict_proba(X[t == 0])[:, 1] - y[t == 0]

        self.tau_t = LGBMRegressor(**PARAMS).fit(X[t == 1], effect_treated)
        self.tau_c = LGBMRegressor(**PARAMS).fit(X[t == 0], effect_control)
        return self

    def predict_uplift(self, X):
        return self.p * self.tau_c.predict(X) + (1 - self.p) * self.tau_t.predict(X)