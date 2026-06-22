import numpy as np
import pytest

from mlforge.cli import main
from mlforge.config import Settings
from mlforge.data import DATASETS, make_dataset


def test_datasets_listed():
    assert set(DATASETS) == {"highdim", "lowdim", "null"}


def test_make_dataset_shapes_and_split():
    ds = make_dataset("highdim", Settings(seed=0))
    assert ds.X_train.shape[0] == ds.y_train.shape[0]
    assert ds.X_oracle.shape[0] > ds.X_train.shape[0]  # oracle is large
    assert ds.X_train.shape[1] == ds.X_oracle.shape[1] == ds.n_features


def test_make_dataset_deterministic():
    a = make_dataset("highdim", Settings(seed=3))
    b = make_dataset("highdim", Settings(seed=3))
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_train, b.y_train)
    # Different seeds give different data.
    c = make_dataset("highdim", Settings(seed=4))
    assert not np.array_equal(a.X_train, c.X_train)


def test_null_is_chance_and_signal_is_not():
    assert make_dataset("null", Settings(seed=0)).bayes_accuracy == 0.5
    assert make_dataset("highdim", Settings(seed=0)).bayes_accuracy > 0.6


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        make_dataset("nope")


def test_cli_data_command(capsys):
    assert main(["data", "--dataset", "lowdim", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "Bayes-optimal accuracy" in out


def test_cli_select_command(capsys):
    # leaky is fast (no nested loop) -> good for a smoke test.
    assert main(["select", "--protocol", "leaky", "--dataset", "highdim", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "optimism" in out and "reported" in out
