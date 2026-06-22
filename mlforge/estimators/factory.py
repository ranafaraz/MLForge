"""The model zoo and the search grid the forge selects over.

The grid is deliberately wide -- several estimators, several hyperparameters each,
and several feature-count choices -- because best-of-grid optimism (selection
bias) grows with the number of configurations tried. A tiny grid would hide the
very effect this project measures.
"""

from __future__ import annotations

from mlforge.estimators.base import BaseComponent
from mlforge.estimators.knn import KNNClassifier
from mlforge.estimators.logistic import LogisticRegression
from mlforge.estimators.naive_bayes import GaussianNB
from mlforge.types import Config

_REGISTRY = {
    "logistic": LogisticRegression,
    "knn": KNNClassifier,
    "gaussian_nb": GaussianNB,
}

# Per-estimator hyperparameter grids. Deliberately wide -- a realistic AutoML
# search tries many configurations, and best-of-grid selection bias grows with the
# number of candidates. A tiny grid would hide the very effect we measure.
_PARAM_GRID = {
    "logistic": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "knn": [{"k": k} for k in (1, 3, 7, 15)],
    "gaussian_nb": [{"var_smoothing": s} for s in (1e-9, 1e-3)],
}

# Feature counts the univariate selector is allowed to keep.
K_FEATURES = (5, 10, 20, 40)


def make_estimator(name: str, params: dict | None = None) -> BaseComponent:
    if name not in _REGISTRY:
        raise ValueError(f"unknown estimator {name!r}; choose from {tuple(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))


def build_grid(max_features: int) -> list[Config]:
    """Every (estimator, hyperparameters, k) combination, with ``k`` capped at the
    number of features actually available so low-dimensional data gets a smaller,
    honestly-less-overfittable grid."""

    ks = [k for k in K_FEATURES if k < max_features] + [min(max_features, max(K_FEATURES))]
    ks = sorted(set(ks))
    grid: list[Config] = []
    for est, param_list in _PARAM_GRID.items():
        for params in param_list:
            for k in ks:
                grid.append(Config(estimator=est, params=dict(params), k_features=k))
    return grid


# A static reference grid (assuming plenty of features) for introspection/tests.
GRID = build_grid(max_features=max(K_FEATURES))
