"""Cross-validation primitives: stratified folds and a generic CV scorer.

Everything here takes a ``build`` thunk that returns a *fresh, unfitted* model so
no state survives across folds. That single discipline is what separates an
honest estimate from a leaky one -- the protocols in :mod:`mlforge.selection`
differ only in *what* they hand to these functions, never in the splitting logic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mlforge.metrics import accuracy_score


def stratified_folds(y, n_folds: int, rng: np.random.Generator):
    """Return ``[(train_idx, test_idx), ...]`` with class balance preserved.

    Each class's shuffled indices are dealt round-robin into the folds so every
    fold sees roughly the same class proportions -- important on the small splits
    here where an unstratified fold could end up nearly single-class.
    """

    y = np.asarray(y)
    n = len(y)
    fold_of = np.empty(n, dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    folds = []
    for f in range(n_folds):
        test_idx = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        folds.append((train_idx, test_idx))
    return folds


def cross_val_score(
    build: Callable[[], object],
    X,
    y,
    folds,
) -> list[float]:
    """Fit a fresh model per fold and score it on the held-out fold."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scores = []
    for train_idx, test_idx in folds:
        model = build()
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return scores
