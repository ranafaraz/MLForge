"""Minimal end-to-end example: forge a model three ways on one dataset and show
how the reported score drifts away from the oracle truth as malpractice creeps in.

Run: ``python examples/run_forge.py``
"""

from __future__ import annotations

from mlforge.config import Settings
from mlforge.data import make_dataset
from mlforge.selection import PROTOCOLS, run_protocol


def main() -> None:
    settings = Settings(dataset="highdim", seed=0)
    ds = make_dataset(settings.dataset, settings)
    print(f"dataset={ds.name}  train={ds.n_train}x{ds.n_features}  Bayes={ds.bayes_accuracy:.3f}")
    print(f"{'protocol':<10}{'reported':>10}{'oracle':>9}{'optimism':>10}")
    for proto in PROTOCOLS:
        r = run_protocol(proto, ds, settings)
        print(f"{proto:<10}{r.reported_score:>10.3f}{r.oracle_score:>9.3f}{r.optimism:>+10.3f}")
    print("\nThe leaky protocol reports the rosiest number and is the most wrong;")
    print("nested cross-validation reports a number you can actually trust.")


if __name__ == "__main__":
    main()
