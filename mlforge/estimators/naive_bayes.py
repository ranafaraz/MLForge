"""Gaussian naive Bayes with a variance-smoothing hyperparameter."""

from __future__ import annotations

import numpy as np

from mlforge.estimators.base import BaseComponent


class GaussianNB(BaseComponent):
    _param_names = ("var_smoothing",)

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        eps = self.var_smoothing * float(X.var(axis=0).max() + 1e-12)
        self.theta_, self.var_, self.prior_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            if len(Xc) == 0:
                # Degenerate fold: fall back to global stats / uniform prior.
                Xc = X
            self.theta_.append(Xc.mean(axis=0))
            self.var_.append(Xc.var(axis=0) + eps)
            self.prior_.append(max(len(X[y == c]), 1) / len(X))
        self.theta_ = np.array(self.theta_)
        self.var_ = np.array(self.var_)
        self.prior_ = np.array(self.prior_)
        return self

    def _log_joint(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for i in range(len(self.classes_)):
            ll = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[i]) + (X - self.theta_[i]) ** 2 / self.var_[i],
                axis=1,
            )
            out.append(np.log(self.prior_[i]) + ll)
        return np.vstack(out).T

    def predict(self, X):
        return self.classes_[np.argmax(self._log_joint(X), axis=1)]
