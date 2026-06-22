"""Deterministic synthetic classifier data with a *known* generative process.

Every dataset bundles a small train split and a large held-out ``oracle`` split
drawn from the same distribution. Because we know the data-generating logit we
also know the Bayes-optimal accuracy, so the oracle accuracy of a fitted model is
checkable against a real ceiling -- no downloads, no labels to trust.

Three regimes drive the whole story:

* ``highdim`` -- many features, few informative. Univariate feature selection has
  a huge surface to overfit, so fitting it on all the data (leakage) inflates the
  reported score and best-of-grid selection adds more optimism.
* ``lowdim`` -- few features, all informative. There is nothing to overfit in
  feature selection, so the honest and leaky protocols *agree*: this is the
  control that proves leakage needs a high-dimensional selection step to bite.
* ``null`` -- labels are independent of the features. True accuracy is 0.5, so any
  reported signal above chance is a pure artifact of the evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlforge.types import Dataset

N_TRAIN = 160
N_ORACLE = 4000


@dataclass(frozen=True)
class Regime:
    name: str
    n_features: int
    n_informative: int
    signal: float  # scales the logit; larger => higher Bayes accuracy
    random_labels: bool = False


REGIMES = {
    "highdim": Regime("highdim", n_features=200, n_informative=6, signal=1.05),
    "lowdim": Regime("lowdim", n_features=8, n_informative=8, signal=0.62),
    "null": Regime("null", n_features=200, n_informative=0, signal=0.0, random_labels=True),
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_regime(regime: Regime, rng: np.random.Generator) -> Dataset:
    n_total = N_TRAIN + N_ORACLE
    p = regime.n_features
    X = rng.standard_normal((n_total, p))

    if regime.random_labels:
        y = rng.integers(0, 2, size=n_total)
        bayes = 0.5
    else:
        # The first ``n_informative`` columns carry signed unit weights; the rest
        # are pure noise. Label noise comes from the logistic link, so the Bayes
        # ceiling is well below 1.0 and the task is honestly hard.
        w = rng.choice(np.array([-1.0, 1.0]), size=regime.n_informative)
        logit = regime.signal * (X[:, : regime.n_informative] @ w)
        prob = _sigmoid(logit)
        y = (rng.random(n_total) < prob).astype(int)
        # Bayes accuracy = E[max(p, 1-p)] under the true posterior.
        bayes = float(np.mean(np.maximum(prob, 1.0 - prob)))

    return Dataset(
        name=regime.name,
        X_train=X[:N_TRAIN],
        y_train=y[:N_TRAIN],
        X_oracle=X[N_TRAIN:],
        y_oracle=y[N_TRAIN:],
        n_informative=regime.n_informative,
        bayes_accuracy=bayes,
    )
