import numpy as np
import pytest

from mlforge.estimators import GaussianNB, KNNClassifier, LogisticRegression, clone
from mlforge.estimators.factory import GRID, build_grid, make_estimator
from mlforge.metrics import accuracy_score


def _easy_problem(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "est",
    [LogisticRegression(), KNNClassifier(k=5), GaussianNB()],
)
def test_estimators_learn_separable(est):
    X, y = _easy_problem()
    est.fit(X, y)
    acc = accuracy_score(y, est.predict(X))
    assert acc >= 0.9


def test_predict_shape_and_labels():
    X, y = _easy_problem()
    pred = LogisticRegression().fit(X, y).predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})


def test_clone_is_fresh_same_params():
    est = LogisticRegression(C=0.25, n_iter=42)
    X, y = _easy_problem()
    est.fit(X, y)
    fresh = clone(est)
    assert fresh.get_params() == {"C": 0.25, "lr": est.lr, "n_iter": 42}
    # A clone carries no fitted state.
    assert not hasattr(fresh, "coef_")


def test_set_params_rejects_unknown():
    with pytest.raises(ValueError):
        LogisticRegression().set_params(nonsense=1)


def test_knn_k_capped_to_n_samples():
    X = np.zeros((3, 2))
    y = np.array([0, 1, 1])
    pred = KNNClassifier(k=99).fit(X, y).predict(X)
    # All points identical -> majority class (1) wins for every prediction.
    assert list(pred) == [1, 1, 1]


def test_make_estimator_and_grid():
    assert isinstance(make_estimator("logistic", {"C": 2.0}), LogisticRegression)
    with pytest.raises(ValueError):
        make_estimator("nope")
    # The grid caps k at the available feature count.
    small = build_grid(max_features=6)
    assert all(c.k_features <= 6 for c in small)
    assert len(GRID) > len(small)
