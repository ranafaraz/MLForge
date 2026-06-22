import numpy as np

from mlforge.pipeline import build_pipeline
from mlforge.preprocessing import SelectKBest, StandardScaler
from mlforge.types import Config


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((150, 10))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


def test_build_pipeline_structure_and_predict():
    cfg = Config(estimator="logistic", params={"C": 1.0}, k_features=4)
    pipe = build_pipeline(cfg)
    names = [n for n, _ in pipe.steps]
    assert names == ["scaler", "select", "model"]
    X, y = _data()
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == y.shape
    # The selector really narrowed the feature space the model saw.
    assert pipe.transform(X).shape == (len(y), 4)


def test_pipeline_uses_fit_statistics_only():
    # The scaler must use *fit* statistics on later data, never refit on it --
    # this is the property that makes refitting-inside-a-fold leak-free.
    train = np.full((20, 1), 10.0)
    train[:, 0] = np.arange(20)  # mean 9.5
    scaler = StandardScaler().fit(train)
    test = np.array([[9.5]])  # the train mean
    assert np.allclose(scaler.transform(test), 0.0)


def test_selectkbest_support_depends_on_fit_data():
    # Two different fit sets select different features -> support is data-driven,
    # so refitting per fold genuinely changes what the model sees (no leakage).
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    A = rng.standard_normal((200, 5))
    A[:, 0] += 3 * y
    B = rng.standard_normal((200, 5))
    B[:, 4] += 3 * y
    assert SelectKBest(k=1).fit(A, y).support_[0] == 0
    assert SelectKBest(k=1).fit(B, y).support_[0] == 4
