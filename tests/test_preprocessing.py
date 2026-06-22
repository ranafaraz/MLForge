import numpy as np

from mlforge.preprocessing import SelectKBest, StandardScaler


def test_standard_scaler_standardizes():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(100, 3))
    Xt = StandardScaler().fit(X).transform(X)
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xt.std(axis=0), 1.0, atol=1e-9)


def test_standard_scaler_constant_column_safe():
    X = np.ones((10, 2))
    X[:, 1] = np.arange(10)
    Xt = StandardScaler().fit(X).transform(X)
    assert np.all(np.isfinite(Xt))  # zero-variance column must not divide by zero


def test_selectkbest_picks_informative():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, 6))
    # Features 0 and 3 carry the label; the rest are noise.
    X[:, 0] += 2.0 * y
    X[:, 3] -= 2.0 * y
    sel = SelectKBest(k=2).fit(X, y)
    assert set(sel.support_.tolist()) == {0, 3}
    assert sel.transform(X).shape == (n, 2)


def test_selectkbest_caps_k():
    X = np.random.default_rng(2).standard_normal((20, 3))
    y = np.array([0, 1] * 10)
    sel = SelectKBest(k=10).fit(X, y)
    assert sel.transform(X).shape[1] == 3
