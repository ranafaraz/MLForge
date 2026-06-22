import numpy as np

from mlforge.cv import cross_val_score, stratified_folds
from mlforge.estimators import LogisticRegression


def test_stratified_folds_partition_and_balance():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 40)
    folds = stratified_folds(y, 5, rng)
    assert len(folds) == 5
    seen = []
    for train_idx, test_idx in folds:
        # train and test are disjoint and together cover everything.
        assert set(train_idx).isdisjoint(set(test_idx))
        assert sorted(set(train_idx) | set(test_idx)) == list(range(len(y)))
        seen.extend(test_idx.tolist())
        # Each fold keeps roughly the 60/40 class balance.
        frac1 = y[test_idx].mean()
        assert 0.25 <= frac1 <= 0.55
    # Every sample is tested exactly once.
    assert sorted(seen) == list(range(len(y)))


def test_stratified_folds_deterministic():
    y = np.array([0, 1] * 50)
    a = stratified_folds(y, 4, np.random.default_rng(7))
    b = stratified_folds(y, 4, np.random.default_rng(7))
    for (ta, ea), (tb, eb) in zip(a, b, strict=True):
        assert np.array_equal(ea, eb) and np.array_equal(ta, tb)


def test_cross_val_score_runs_per_fold():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 3))
    y = (X[:, 0] > 0).astype(int)
    folds = stratified_folds(y, 4, rng)
    scores = cross_val_score(lambda: LogisticRegression(), X, y, folds)
    assert len(scores) == 4
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert np.mean(scores) >= 0.9
