"""Aggregate forge results across seeds into mean reported / oracle / optimism."""

from __future__ import annotations

from statistics import mean, pstdev

from mlforge.types import ForgeResult


def aggregate(results: list[ForgeResult]) -> dict:
    """Pool a protocol's results over many seeds of one dataset."""
    rep = [r.reported_score for r in results]
    orc = [r.oracle_score for r in results]
    opt = [r.optimism for r in results]
    return {
        "n": len(results),
        "reported": mean(rep),
        "oracle": mean(orc),
        "optimism": mean(opt),
        "optimism_sd": pstdev(opt) if len(opt) > 1 else 0.0,
    }
