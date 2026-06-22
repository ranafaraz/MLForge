"""Cross-check the optimism decomposition against scikit-learn (optional).

Skipped unless scikit-learn is installed (`pip install -e ".[sklearn]"`). It does
not assert exact numeric equality with the from-scratch core -- different solvers
land in slightly different places -- but it must reproduce the same *story*: leaky
is optimistic, nested is honest, the null signal collapses.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from mlforge.config import Settings  # noqa: E402
from mlforge.data import make_dataset  # noqa: E402
from mlforge.sklearn_check import run_protocol_sklearn  # noqa: E402

FAST = dict(folds=4, inner_folds=2)


def test_sklearn_reproduces_the_dissociation():
    s = Settings(seed=0, **FAST)
    ds = make_dataset("highdim", s)
    leaky = run_protocol_sklearn("leaky", ds, s)
    nested = run_protocol_sklearn("nested", ds, s)
    assert leaky.optimism > nested.optimism
    assert abs(nested.optimism) < 0.12


def test_sklearn_null_collapses():
    s = Settings(seed=0, **FAST)
    ds = make_dataset("null", s)
    leaky = run_protocol_sklearn("leaky", ds, s)
    nested = run_protocol_sklearn("nested", ds, s)
    assert leaky.reported_score > nested.reported_score + 0.05
