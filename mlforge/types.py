"""Core data types passed between the data, selection and eval layers.

These are plain dataclasses; the numeric payloads are numpy arrays. Keeping them
explicit (rather than passing bare tuples around) is what lets the forge, the CLI
and the eval harness all speak the same language about a *result*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """A train split plus a large held-out *oracle* test split.

    The oracle is drawn from the same generative process as the train split and
    is big enough that accuracy on it is a low-variance estimate of true
    generalization. It is the ground truth every protocol's *reported* score is
    measured against -- it is never shown to the forge during selection.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_oracle: np.ndarray
    y_oracle: np.ndarray
    n_informative: int
    bayes_accuracy: float  # accuracy of the data-generating optimal classifier

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])


@dataclass
class Config:
    """One point in the search grid: an estimator name with its hyperparameters
    and the number of univariate-selected features to feed it."""

    estimator: str
    params: dict
    k_features: int

    def label(self) -> str:
        ps = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.estimator}(k={self.k_features};{ps})"


@dataclass
class ForgeResult:
    """The outcome of running one selection protocol on one dataset.

    ``reported_score`` is what the protocol would put in a paper/README; the
    forge then ships the chosen pipeline (refit on all training data) and we score
    it on the oracle to get ``oracle_score``. The headline number is ``optimism``.
    """

    protocol: str
    dataset: str
    reported_score: float
    oracle_score: float
    chosen: Config
    folds: list[float] = field(default_factory=list)

    @property
    def optimism(self) -> float:
        """Reported minus truth: how much the protocol oversold the model."""
        return self.reported_score - self.oracle_score
