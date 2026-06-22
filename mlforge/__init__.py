"""MLForge: a leakage-safe tabular model-selection forge.

Proper pipelining buys a preprocessing-leakage correction; nested cross-validation
buys a selection-bias correction. Both are *measured* against a held-out oracle,
with a random-label null where the leaky protocol manufactures fake signal.
"""

from __future__ import annotations

# Pin BLAS to a single thread *before* numpy is imported. The whole benchmark is
# thousands of tiny matrix ops (small folds x small grids); with multi-threaded
# BLAS the per-call thread-pool overhead dominates and contention can blow a few
# seconds up to several minutes -- which would make CI unusable. One thread is
# both faster here and fully deterministic. setdefault so an explicit env wins.
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from mlforge.config import Settings
from mlforge.data import DATASETS, make_dataset
from mlforge.estimators import GRID, clone, make_estimator
from mlforge.metrics import accuracy_score
from mlforge.pipeline import Pipeline, build_pipeline
from mlforge.selection import PROTOCOLS, run_protocol
from mlforge.types import Config, Dataset, ForgeResult

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "make_dataset",
    "DATASETS",
    "make_estimator",
    "clone",
    "GRID",
    "accuracy_score",
    "Pipeline",
    "build_pipeline",
    "run_protocol",
    "PROTOCOLS",
    "Config",
    "Dataset",
    "ForgeResult",
]
