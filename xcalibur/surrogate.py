"""A learned ranker for measured search: a bootstrap ensemble of LightGBM regressors and an upper confidence bound.

Each member is trained on a bootstrap resample of the measured designs; the spread of their predictions is a cheap
uncertainty estimate. Ranking by ``mu + kappa * sd`` spends measurements both where the model expects a gain and
where it is unsure. The ranker only orders what to measure; it never decides what is faster.

Useful features in our setting were exact and cheap: field one-hots, the bits of a decoder graph's code (which let
the model generalize to graphs nobody has measured), compiled SASS counts, and quantities derived by the WarpSpeed
checker (register cap, warps per scheduler, bytes in flight).
"""
from __future__ import annotations

import numpy as np


class Surrogate:
  def __init__(self, members: int = 10, **lgbm):
    self.members = members
    self.params = dict(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=3,
                           colsample_bytree=0.8, subsample=0.9, subsample_freq=1, verbose=-1) | lgbm
    self.models = []

  def fit(self, X, y, w=None):
    import lightgbm as lgb                      # noqa: PLC0415
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    self.models = []
    for s in range(self.members):
      i = np.random.default_rng(s).integers(0, len(y), len(y))
      m = lgb.LGBMRegressor(random_state=s, **self.params)
      m.fit(X[i], y[i], sample_weight=w[i])
      self.models.append(m)
    return self

  def predict(self, X):
    X = np.asarray(X, float)
    P = np.stack([m.predict(X) for m in self.models])
    return P.mean(0), P.std(0) + 1e-6

  def ucb(self, X, kappa: float = 1.0, baseline: float = 0.0):
    mu, sd = self.predict(X)
    return mu + kappa * sd - baseline, mu, sd

  def top(self, X, k: int, kappa: float = 1.0, baseline: float = 0.0):
    score, mu, sd = self.ucb(X, kappa, baseline)
    order = np.argsort(-score)[:k]
    return [(int(i), float(mu[i]), float(sd[i])) for i in order]


def bits(code: int, n: int = 32):
  """Binary features of an integer code (e.g. a decoder graph's id)."""
  return [float((code >> i) & 1) for i in range(n)]


def one_hot(value, choices):
  return [float(value == c) for c in choices]
