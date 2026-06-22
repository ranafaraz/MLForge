"""Minimal scikit-learn-style component base.

Every estimator and transformer exposes ``get_params``/``set_params`` and a
``fit(X, y)`` that returns ``self``. That uniformity is what lets :func:`clone`
produce a fresh, unfitted copy with the same hyperparameters -- the single most
important primitive for honest cross-validation, because each fold must train a
brand-new model rather than silently reuse state fitted on other folds' data.
"""

from __future__ import annotations

import copy


class BaseComponent:
    """Shared get/set-params plumbing. Subclasses store hyperparameters as
    plain instance attributes named exactly as in ``_param_names``."""

    _param_names: tuple[str, ...] = ()

    def get_params(self) -> dict:
        return {name: getattr(self, name) for name in self._param_names}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self._param_names:
                raise ValueError(f"{type(self).__name__} has no parameter {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):  # pragma: no cover - overridden
        raise NotImplementedError


def clone(component: BaseComponent) -> BaseComponent:
    """Return a fresh, unfitted component with identical hyperparameters.

    Uses the registered hyperparameters only, so fitted attributes (weights,
    stored training rows, selected feature masks) are intentionally dropped.
    """

    fresh = type(component)()
    fresh.set_params(**copy.deepcopy(component.get_params()))
    return fresh
