import numpy as np

from mlforge.config import Settings
from mlforge.data import make_dataset
from mlforge.selection import PROTOCOLS, run_protocol

# Smaller fold counts keep the test fast while preserving the structure.
FAST = dict(folds=4, inner_folds=2)


def test_protocols_run_and_return_valid_results():
    s = Settings(seed=0, **FAST)
    ds = make_dataset("highdim", s)
    for proto in PROTOCOLS:
        r = run_protocol(proto, ds, s)
        assert r.protocol == proto
        assert 0.0 <= r.reported_score <= 1.0
        assert 0.0 <= r.oracle_score <= 1.0
        assert r.chosen.estimator in {"logistic", "knn", "gaussian_nb"}


def test_pipeline_and_nested_ship_the_same_model():
    # By design they select identically on all data, so their oracle (truth of the
    # shipped model) must match -- which is what makes their reported-score gap a
    # clean measure of selection bias alone.
    s = Settings(seed=1, **FAST)
    ds = make_dataset("highdim", s)
    rp = run_protocol("pipeline", ds, s)
    rn = run_protocol("nested", ds, s)
    assert rp.chosen.label() == rn.chosen.label()
    assert rp.oracle_score == rn.oracle_score


def test_leaky_is_more_optimistic_than_nested():
    # Averaged over a few seeds to avoid single-seed noise.
    opt_leaky, opt_nested = [], []
    for seed in range(3):
        s = Settings(seed=seed, **FAST)
        ds = make_dataset("highdim", s)
        opt_leaky.append(run_protocol("leaky", ds, s).optimism)
        opt_nested.append(run_protocol("nested", ds, s).optimism)
    assert np.mean(opt_leaky) > np.mean(opt_nested) + 0.03
    assert abs(np.mean(opt_nested)) < 0.10  # nested stays close to the truth


def test_null_leaky_manufactures_signal():
    # On random labels the leaky protocol still reports accuracy well above the
    # 0.5 truth; its oracle (the shipped model on held-out data) is ~chance.
    s = Settings(seed=0, **FAST)
    ds = make_dataset("null", s)
    assert ds.bayes_accuracy == 0.5
    leaky = run_protocol("leaky", ds, s)
    assert leaky.reported_score >= 0.58
    assert leaky.oracle_score <= 0.56
    assert leaky.optimism >= 0.05


def test_unknown_protocol_raises():
    s = Settings(**FAST)
    ds = make_dataset("lowdim", s)
    try:
        run_protocol("bogus", ds, s)
        raised = False
    except ValueError:
        raised = True
    assert raised
