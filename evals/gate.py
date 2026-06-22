"""CI quality gate: fail the build if the leakage story regresses.

Like the rest of the portfolio, the gate enforces the *shape* of the result, not
just that a number is high. It checks the ordered dissociation (leaky more
optimistic than pipeline more optimistic than nested), that nested lands close to
the oracle truth, that the low-dimensional control makes the protocols agree, and
that the random-label null collapses (the leaky protocol manufactures a signal
nested CV refuses to report). Floors sit well below the observed offline numbers
so ordinary seed variation never flakes CI, while a real regression -- a leak that
creeps back in, a broken nested loop -- trips it. No API keys, no downloads.
"""

from __future__ import annotations

import sys

from evals.harness import run_eval


def _checks(r: dict) -> list[tuple[str, bool, str]]:
    order_ok = (
        r["hd_leaky_reported"] >= r["hd_pipeline_reported"] + 0.02
        and r["hd_pipeline_reported"] >= r["hd_nested_reported"] + 0.005
    )
    null_collapse = r["null_leaky_reported"] - r["null_nested_reported"]
    return [
        (
            "nested is honest on highdim (reported ~ oracle)",
            abs(r["hd_nested_opt"]) <= 0.05,
            f"|nested optimism|={abs(r['hd_nested_opt']):.3f} <= 0.05",
        ),
        (
            "leaky is badly optimistic on highdim",
            r["hd_leaky_opt"] >= 0.08,
            f"leaky optimism={r['hd_leaky_opt']:+.3f} >= 0.08",
        ),
        (
            "proper pipelining buys a preprocessing-leakage correction",
            r["hd_leakage_effect"] >= 0.04,
            f"leakage effect={r['hd_leakage_effect']:+.3f} >= 0.04 "
            f"(opt leaky {r['hd_leaky_opt']:+.3f} -> pipeline {r['hd_pipeline_opt']:+.3f})",
        ),
        (
            "nested CV buys a selection-bias correction",
            r["hd_selection_effect"] >= 0.012,
            f"selection effect={r['hd_selection_effect']:+.3f} >= 0.012 "
            f"(opt pipeline {r['hd_pipeline_opt']:+.3f} -> nested {r['hd_nested_opt']:+.3f})",
        ),
        (
            "reported scores are ordered leaky > pipeline > nested",
            order_ok,
            f"reported L/P/N = {r['hd_leaky_reported']:.3f}/"
            f"{r['hd_pipeline_reported']:.3f}/{r['hd_nested_reported']:.3f}",
        ),
        (
            "control (lowdim): nothing to leak, leaky stays honest",
            r["low_leaky_opt"] <= 0.05,
            f"lowdim leaky optimism={r['low_leaky_opt']:+.3f} <= 0.05",
        ),
        (
            "control (lowdim): protocols agree",
            abs(r["low_leaky_opt"] - r["low_nested_opt"]) <= 0.04,
            f"|leaky-nested optimism|={abs(r['low_leaky_opt'] - r['low_nested_opt']):.3f} <= 0.04",
        ),
        (
            "null: leaky manufactures signal above the 0.5 truth",
            r["null_leaky_reported"] >= 0.58 and r["null_leaky_oracle"] <= 0.55,
            f"null leaky reported={r['null_leaky_reported']:.3f} >= 0.58, "
            f"oracle={r['null_leaky_oracle']:.3f} <= 0.55",
        ),
        (
            "null: nested refuses the fake signal (collapse)",
            r["null_nested_reported"] <= 0.57 and null_collapse >= 0.05,
            f"null nested reported={r['null_nested_reported']:.3f} <= 0.57, "
            f"collapse={null_collapse:+.3f} >= 0.05",
        ),
    ]


def main() -> int:
    res = run_eval()
    checks = _checks(res)
    print("MLForge eval gate")
    failures = []
    for desc, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: {detail}")
        if not ok:
            failures.append(desc)
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
