"""The forge: model selection under three evaluation protocols.

All three search the *same* grid over the *same* data and ship their chosen model
the same way (refit on all training data, then score on the oracle). They differ
in exactly one structural choice each, which is the whole experiment:

* ``leaky``    -- fit the scaler + feature selector once on *all* training data,
                  then flat-CV the estimator on the already-selected features.
                  Two sins at once: preprocessing leakage *and* best-of-grid
                  selection bias.
* ``pipeline`` -- refit the whole pipeline (scaler + selector + estimator) inside
                  every fold, so no preprocessing leakage; but still report the
                  best-of-grid mean, so selection bias remains.
* ``nested``   -- nested CV: an inner loop selects the config, an outer loop
                  estimates generalization. Neither sin. The honest reference.

The reported score is what the protocol would advertise; ``oracle_score`` is the
truth of the shipped model. The gap between them -- *optimism* -- is the headline,
and the difference of gaps across protocols isolates each individual effect.
"""

from __future__ import annotations

import numpy as np

from mlforge.config import Settings
from mlforge.cv import cross_val_score, stratified_folds
from mlforge.estimators.factory import build_grid, make_estimator
from mlforge.metrics import accuracy_score
from mlforge.pipeline import build_pipeline
from mlforge.preprocessing import SelectKBest, StandardScaler
from mlforge.types import Config, Dataset, ForgeResult

PROTOCOLS = ("leaky", "pipeline", "nested")


def _oracle_score(config: Config, ds: Dataset) -> float:
    """Ship the chosen config (refit honestly on all training data) and score it
    on the held-out oracle -- the ground-truth generalization."""
    model = build_pipeline(config).fit(ds.X_train, ds.y_train)
    return accuracy_score(ds.y_oracle, model.predict(ds.X_oracle))


def _leaky_mean(config: Config, ds: Dataset, folds) -> float:
    # Preprocessing fitted on ALL training rows (incl. the held-out folds) -> leak.
    scaler = StandardScaler().fit(ds.X_train, ds.y_train)
    Xs = scaler.transform(ds.X_train)
    sel = SelectKBest(k=config.k_features).fit(Xs, ds.y_train)
    Xsel = sel.transform(Xs)
    scores = cross_val_score(
        lambda: make_estimator(config.estimator, config.params), Xsel, ds.y_train, folds
    )
    return float(np.mean(scores))


def _pipeline_mean(config: Config, ds: Dataset, folds) -> float:
    # Whole pipeline refit inside each fold -> preprocessing sees train rows only.
    scores = cross_val_score(lambda: build_pipeline(config), ds.X_train, ds.y_train, folds)
    return float(np.mean(scores))


def _select_flat(score_fn, ds: Dataset, grid, folds) -> tuple[Config, float, list[float]]:
    means = [score_fn(cfg, ds, folds) for cfg in grid]
    best = int(np.argmax(means))
    return grid[best], means[best], means


def _select_inner(ds_X, ds_y, grid, inner_folds) -> Config:
    """Pick the best config by flat pipeline-CV on the supplied (sub)set."""
    best_cfg, best_mean = grid[0], -1.0
    for cfg in grid:
        m = float(
            np.mean(cross_val_score(lambda c=cfg: build_pipeline(c), ds_X, ds_y, inner_folds))
        )
        if m > best_mean:
            best_cfg, best_mean = cfg, m
    return best_cfg


def run_protocol(protocol: str, ds: Dataset, settings: Settings | None = None) -> ForgeResult:
    settings = settings or Settings()
    grid = build_grid(ds.n_features)
    cv_folds = stratified_folds(ds.y_train, settings.folds, settings.rng(101))

    if protocol == "leaky":
        chosen, reported, _ = _select_flat(_leaky_mean, ds, grid, cv_folds)
        folds_used = []
    elif protocol == "pipeline":
        chosen, reported, _ = _select_flat(_pipeline_mean, ds, grid, cv_folds)
        folds_used = []
    elif protocol == "nested":
        outer = stratified_folds(ds.y_train, settings.folds, settings.rng(202))
        outer_scores = []
        for fi, (tr, te) in enumerate(outer):
            inner = stratified_folds(ds.y_train[tr], settings.inner_folds, settings.rng(303, fi))
            best = _select_inner(ds.X_train[tr], ds.y_train[tr], grid, inner)
            model = build_pipeline(best).fit(ds.X_train[tr], ds.y_train[tr])
            outer_scores.append(accuracy_score(ds.y_train[te], model.predict(ds.X_train[te])))
        reported = float(np.mean(outer_scores))
        folds_used = outer_scores
        # Ship the config a flat (honest) selection on all the data would pick --
        # identical to the ``pipeline`` protocol's shipped model, so the two share
        # an oracle score and their reported-score gap isolates selection bias.
        chosen, _, _ = _select_flat(_pipeline_mean, ds, grid, cv_folds)
    else:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")

    return ForgeResult(
        protocol=protocol,
        dataset=ds.name,
        reported_score=reported,
        oracle_score=_oracle_score(chosen, ds),
        chosen=chosen,
        folds=folds_used,
    )
