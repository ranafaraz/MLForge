"""Resolve a dataset name + seed into a reproducible :class:`Dataset`."""

from __future__ import annotations

from mlforge.config import Settings
from mlforge.data.synthetic import REGIMES, make_regime
from mlforge.types import Dataset

DATASETS = tuple(REGIMES.keys())

# Distinct rng offset per regime so seed N gives independent draws across regimes.
_OFFSET = {name: i for i, name in enumerate(REGIMES)}


def make_dataset(name: str, settings: Settings | None = None, seed: int | None = None) -> Dataset:
    if name not in REGIMES:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    settings = settings or Settings()
    if seed is not None:
        settings = Settings(**{**settings.__dict__, "seed": seed})
    rng = settings.rng(_OFFSET[name])
    return make_regime(REGIMES[name], rng)
