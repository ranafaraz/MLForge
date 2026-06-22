"""Transformers: standardisation and univariate feature selection.

``SelectKBest`` is the component the whole leakage story turns on. It uses the
labels to rank features, so *where it is fitted* decides whether information from
the held-out folds bleeds into the model. Fit it once on all the data and the
selected features are partly chosen for their fit to the rows you later "test" on
-- that is the leak. Fit it inside each fold and the leak is gone.
"""

from __future__ import annotations

import numpy as np

from mlforge.estimators.base import BaseComponent


class StandardScaler(BaseComponent):
    """Zero-mean, unit-variance per feature (stats taken from the fit split)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


class SelectKBest(BaseComponent):
    """Keep the ``k`` features with the highest absolute point-biserial
    correlation with the (binary) target."""

    _param_names = ("k",)

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum() + 1e-12)
        self.scores_ = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)
        k = min(self.k, X.shape[1])
        # Stable top-k: sort by score desc, then index asc, for determinism.
        order = sorted(range(X.shape[1]), key=lambda j: (-self.scores_[j], j))
        self.support_ = np.array(sorted(order[:k]))
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]
