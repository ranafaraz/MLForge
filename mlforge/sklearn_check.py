"""Optional scikit-learn cross-check of the three protocols.

This module is *not* imported by the offline core and scikit-learn is not a
required dependency. Its only job is to reproduce the leaky / pipeline / nested
optimism decomposition with scikit-learn's own ``Pipeline``,
``cross_val_score`` and ``StratifiedKFold`` so the from-scratch numbers can be
confirmed against the reference implementations.

Enable with::

    pip install -e ".[sklearn]"
    MLFORGE_BACKEND=sklearn mlforge compare --dataset highdim
"""

from __future__ import annotations

import numpy as np

from mlforge.config import Settings
from mlforge.estimators.factory import build_grid
from mlforge.types import Config, Dataset, ForgeResult


def _make_sklearn_estimator(config: Config):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    if config.estimator == "logistic":
        return LogisticRegression(C=config.params["C"], max_iter=500)
    if config.estimator == "knn":
        return KNeighborsClassifier(n_neighbors=config.params["k"])
    if config.estimator == "gaussian_nb":
        return GaussianNB(var_smoothing=config.params["var_smoothing"])
    raise ValueError(config.estimator)


def _make_pipeline(config: Config):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=min(config.k_features, 10_000))),
            ("model", _make_sklearn_estimator(config)),
        ]
    )


def _oracle(config: Config, ds: Dataset) -> float:
    pipe = _make_pipeline(config)
    pipe.set_params(select__k=min(config.k_features, ds.n_features))
    pipe.fit(ds.X_train, ds.y_train)
    return float(np.mean(pipe.predict(ds.X_oracle) == ds.y_oracle))


def run_protocol_sklearn(
    protocol: str, ds: Dataset, settings: Settings | None = None
) -> ForgeResult:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    settings = settings or Settings()
    grid = build_grid(ds.n_features)
    X, y = ds.X_train, ds.y_train
    cv = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=settings.seed)

    def flat_means(build_score):
        return [build_score(cfg) for cfg in grid]

    if protocol == "leaky":
        def score(cfg):
            Xs = StandardScaler().fit_transform(X)
            k = min(cfg.k_features, ds.n_features)
            Xsel = SelectKBest(f_classif, k=k).fit_transform(Xs, y)  # leak: fit on all data
            return cross_val_score(_make_sklearn_estimator(cfg), Xsel, y, cv=cv).mean()

        means = flat_means(score)
        best = grid[int(np.argmax(means))]
        reported = float(np.max(means))
    elif protocol == "pipeline":
        def score(cfg):
            pipe = _make_pipeline(cfg)
            pipe.set_params(select__k=min(cfg.k_features, ds.n_features))
            return cross_val_score(pipe, X, y, cv=cv).mean()

        means = flat_means(score)
        best = grid[int(np.argmax(means))]
        reported = float(np.max(means))
    elif protocol == "nested":
        outer = StratifiedKFold(
            n_splits=settings.folds, shuffle=True, random_state=settings.seed + 1
        )
        inner = StratifiedKFold(
            n_splits=settings.inner_folds, shuffle=True, random_state=settings.seed + 2
        )
        outer_scores = []
        for tr, te in outer.split(X, y):
            inner_means = []
            for cfg in grid:
                pipe = _make_pipeline(cfg)
                pipe.set_params(select__k=min(cfg.k_features, ds.n_features))
                inner_means.append(cross_val_score(pipe, X[tr], y[tr], cv=inner).mean())
            chosen = grid[int(np.argmax(inner_means))]
            pipe = _make_pipeline(chosen)
            pipe.set_params(select__k=min(chosen.k_features, ds.n_features))
            pipe.fit(X[tr], y[tr])
            outer_scores.append(float(np.mean(pipe.predict(X[te]) == y[te])))
        reported = float(np.mean(outer_scores))
        # Ship what a flat pipeline selection on all data would pick.
        pmeans = []
        for cfg in grid:
            pipe = _make_pipeline(cfg)
            pipe.set_params(select__k=min(cfg.k_features, ds.n_features))
            pmeans.append(cross_val_score(pipe, X, y, cv=cv).mean())
        best = grid[int(np.argmax(pmeans))]
    else:
        raise ValueError(protocol)

    return ForgeResult(
        protocol=protocol,
        dataset=ds.name,
        reported_score=reported,
        oracle_score=_oracle(best, ds),
        chosen=best,
    )
