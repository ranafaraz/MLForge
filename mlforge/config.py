"""Environment-driven configuration.

Everything has a deterministic offline default so ``pytest``, ``evals.harness``
and ``evals.gate`` run green with no API keys and no downloads. The one
independent variable of the whole project is ``protocol`` -- how the model
selection is *evaluated* (``nested`` honest vs ``pipeline`` vs ``leaky``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Fixed salt so seeds map to reproducible datasets across processes. Never use
# hash() for this -- Python hash randomisation would make the benchmark
# non-reproducible run-to-run (a lesson learned the hard way elsewhere).
SALT = 0x5F0F6E  # arbitrary fixed constant ("forge")


def _get(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    """Resolved runtime settings.

    protocol : ``nested`` (default, honest) | ``pipeline`` (no leakage, but flat
        best-of-grid selection bias) | ``leaky`` (preprocessing fit on all data +
        flat selection: the textbook mistake)
    dataset  : ``highdim`` (default) | ``lowdim`` (control) | ``null``
    backend  : ``numpy`` (default, from-scratch) | ``sklearn`` (optional cross-check)
    """

    protocol: str = "nested"
    dataset: str = "highdim"
    backend: str = "numpy"
    seed: int = 0
    folds: int = 6
    inner_folds: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            protocol=_get("MLFORGE_PROTOCOL", cls.protocol),
            dataset=_get("MLFORGE_DATASET", cls.dataset),
            backend=_get("MLFORGE_BACKEND", cls.backend),
            seed=_get_int("MLFORGE_SEED", cls.seed),
            folds=_get_int("MLFORGE_FOLDS", cls.folds),
            inner_folds=_get_int("MLFORGE_INNER_FOLDS", cls.inner_folds),
        )

    def rng(self, *offsets: int):
        """A reproducible numpy Generator seeded from SALT + seed + offsets."""
        import numpy as np

        return np.random.default_rng((SALT + self.seed, *offsets))
