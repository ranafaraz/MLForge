"""A scikit-learn-style pipeline: transformers then a final estimator.

The pipeline is the unit the honest protocols clone and refit *inside* each fold.
Because preprocessing (scaling + feature selection) is bundled into it, refitting
the pipeline refits the preprocessing too -- which is exactly what stops held-out
fold information from leaking through the feature selector.
"""

from __future__ import annotations

import numpy as np

from mlforge.estimators.factory import make_estimator
from mlforge.preprocessing import SelectKBest, StandardScaler
from mlforge.types import Config


class Pipeline:
    """``steps`` is a list of ``(name, component)``; all but the last are
    transformers (``fit``/``transform``), the last is an estimator (``predict``)."""

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        Xt = np.asarray(X, dtype=float)
        for _, comp in self.steps[:-1]:
            comp.fit(Xt, y)
            Xt = comp.transform(Xt)
        self.steps[-1][1].fit(Xt, y)
        return self

    def transform(self, X):
        Xt = np.asarray(X, dtype=float)
        for _, comp in self.steps[:-1]:
            Xt = comp.transform(Xt)
        return Xt

    def predict(self, X):
        return self.steps[-1][1].predict(self.transform(X))


def build_pipeline(config: Config) -> Pipeline:
    """A fresh, unfitted pipeline (scale -> select-k -> estimator) for a config."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(k=config.k_features)),
            ("model", make_estimator(config.estimator, config.params)),
        ]
    )
