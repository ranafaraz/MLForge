"""Command-line interface.

Prints ASCII only (Windows consoles are cp1252); the eval harness is what writes
the rich UTF-8 results file. Commands::

    mlforge data    [--dataset]                 # describe a dataset + Bayes ceiling
    mlforge select  [--protocol --dataset --seed]  # one protocol, one dataset
    mlforge compare [--dataset --seed]          # all three protocols side by side
    mlforge eval                                # full benchmark -> evals/RESULTS.md
"""

from __future__ import annotations

import argparse

from mlforge.config import Settings
from mlforge.data import DATASETS, make_dataset
from mlforge.selection import PROTOCOLS, run_protocol


def _forge(protocol, ds, settings):
    """Dispatch to the numpy core or the optional scikit-learn cross-check."""
    if settings.backend == "sklearn":
        from mlforge.sklearn_check import run_protocol_sklearn

        return run_protocol_sklearn(protocol, ds, settings)
    return run_protocol(protocol, ds, settings)


def _settings(args) -> Settings:
    s = Settings.from_env()
    for attr in ("protocol", "dataset", "seed", "folds"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(s, attr, val)
    return s


def _cmd_data(args) -> int:
    s = _settings(args)
    ds = make_dataset(s.dataset, s)
    print(f"dataset={ds.name} seed={s.seed}")
    print(f"  train={ds.n_train} x {ds.n_features} features, oracle={len(ds.y_oracle)}")
    print(f"  informative features: {ds.n_informative}")
    print(f"  Bayes-optimal accuracy: {ds.bayes_accuracy:.3f}")
    print(f"  train class balance: {float(ds.y_train.mean()):.3f}")
    return 0


def _cmd_select(args) -> int:
    s = _settings(args)
    ds = make_dataset(s.dataset, s)
    r = _forge(s.protocol, ds, s)
    print(f"protocol={r.protocol} dataset={r.dataset} seed={s.seed} backend={s.backend}")
    print(f"  reported (CV)  : {r.reported_score:.3f}")
    print(f"  oracle (truth) : {r.oracle_score:.3f}")
    print(f"  optimism       : {r.optimism:+.3f}")
    print(f"  chosen         : {r.chosen.label()}")
    return 0


def _cmd_compare(args) -> int:
    s = _settings(args)
    ds = make_dataset(s.dataset, s)
    print(f"dataset={ds.name} seed={s.seed}  (Bayes {ds.bayes_accuracy:.3f})")
    print(f"  {'protocol':<10} {'reported':>9} {'oracle':>8} {'optimism':>9}  chosen")
    for proto in PROTOCOLS:
        r = _forge(proto, ds, s)
        print(
            f"  {proto:<10} {r.reported_score:>9.3f} {r.oracle_score:>8.3f} "
            f"{r.optimism:>+9.3f}  {r.chosen.label()}"
        )
    return 0


def _cmd_eval(_args) -> int:
    from evals.harness import main as harness_main

    return harness_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlforge", description="Leakage-safe model-selection forge.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, with_protocol=True):
        sp.add_argument("--dataset", choices=DATASETS, help="dataset regime")
        sp.add_argument("--seed", type=int, help="random seed")
        if with_protocol:
            sp.add_argument("--protocol", choices=PROTOCOLS, help="selection protocol")
        sp.add_argument("--folds", type=int, help="cross-validation folds")

    sp = sub.add_parser("data", help="describe a dataset")
    add_common(sp, with_protocol=False)
    sp.set_defaults(func=_cmd_data)

    sp = sub.add_parser("select", help="run one protocol on one dataset")
    add_common(sp)
    sp.set_defaults(func=_cmd_select)

    sp = sub.add_parser("compare", help="all protocols side by side")
    add_common(sp, with_protocol=False)
    sp.set_defaults(func=_cmd_compare)

    sp = sub.add_parser("eval", help="run the full offline benchmark")
    sp.set_defaults(func=_cmd_eval)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
