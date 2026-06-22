"""L2-regularized logistic regression trained by full-batch gradient descent."""

from __future__ import annotations

import numpy as np

from mlforge.estimators.base import BaseComponent


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticRegression(BaseComponent):
    """Binary logistic regression.

    ``C`` is the inverse L2 strength (smaller C = stronger regularization), matching
    the scikit-learn convention so the optional sklearn cross-check lines up.
    """

    _param_names = ("C", "lr", "n_iter")

    def __init__(self, C: float = 1.0, lr: float = 0.7, n_iter: int = 150):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = 0.0
        # Penalty per-sample so lr/C effects are comparable across split sizes.
        lam = 1.0 / (self.C * n)
        for _ in range(self.n_iter):
            z = X @ self.coef_ + self.intercept_
            err = _sigmoid(z) - y
            grad_w = X.T @ err / n + lam * self.coef_
            grad_b = float(np.mean(err))
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
