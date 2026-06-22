"""k-nearest-neighbours classifier (Euclidean, majority vote)."""

from __future__ import annotations

import numpy as np

from mlforge.estimators.base import BaseComponent


class KNNClassifier(BaseComponent):
    _param_names = ("k",)

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y=None):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # Pairwise squared Euclidean distances via the (a-b)^2 expansion.
        sq = (
            (X**2).sum(1)[:, None]
            - 2 * X @ self.X_.T
            + (self.X_**2).sum(1)[None, :]
        )
        k = min(self.k, self.X_.shape[0])
        nn = np.argpartition(sq, kth=k - 1, axis=1)[:, :k]
        votes = self.y_[nn]
        # Mean of 0/1 labels >= 0.5 -> class 1 (ties break to 1, deterministically).
        return (votes.mean(axis=1) >= 0.5).astype(int)
