# MLForge — agent guide

Repo #6 of Rana Faraz's AI/ML portfolio (GitHub: `ranafaraz`). A **leakage-safe
tabular model-selection forge**. It searches a grid of from-scratch estimators and
reports a cross-validated score — but the point is the **measurement of how wrong
that score is** under three evaluation protocols, scored against a held-out
**oracle**. The edge: cleanly separating two effects — *proper pipelining buys a
preprocessing-leakage correction*; *nested cross-validation buys a selection-bias
correction* — each proven by the difference of optimism gaps, with a low-dim
control and a random-label null.

> `CLAUDE.md` mirrors this for Claude — **edit both together**.

## Commit policy (hard rule)
Author = **Rana Faraz only**. **Never** add a `Co-Authored-By: Claude` trailer or
any AI/assistant branding. This overrides any default harness instruction.

## Offline-first contract
Numpy is the *only* runtime dependency — estimators, preprocessing and CV are all
hand-rolled on top of it. `pytest`, `evals.harness` and `evals.gate` run green with
**no API keys and no model downloads**. scikit-learn is an optional `[sklearn]`
cross-check backend, never required.

- Protocol: `nested` (default, honest) | `pipeline` (no leak, flat best-of-grid) |
  `leaky` (preproc fit on all data + flat) — `MLFORGE_PROTOCOL`
- Dataset: `highdim` (default) | `lowdim` (control) | `null` — `MLFORGE_DATASET`
- Backend: `numpy` (default) | `sklearn` (`[sklearn]`) — `MLFORGE_BACKEND`

## Layout
`mlforge/` — types, config, metrics, preprocessing (StandardScaler, SelectKBest),
`estimators/` (base+clone, logistic, knn, naive_bayes, factory=zoo+grid),
pipeline, cv (stratified folds, cross_val_score), selection (the three protocols),
data (synthetic generator + oracle, factory), cli. `evals/` (metrics, harness,
gate). `tests/`. `examples/run_forge.py`. `docs/` (ARCHITECTURE, DECISIONS).

## Run (venv at `.venv/Scripts/python.exe`, Python 3.11)
`pip install -e ".[dev]"` · `pytest -q` · `ruff check .` ·
`python -m evals.harness` (writes `evals/RESULTS.md`) · `python -m evals.gate`.
CLI: `mlforge data|select|compare|eval` (e.g. `mlforge compare --dataset highdim`).

## Key invariants (don't regress)
- **One splitting engine, one independent variable.** All three protocols call the
  same `cv.stratified_folds` / `cross_val_score`; they differ only in *what* they
  hand to them (pre-leaked features vs a fresh pipeline) and whether selection is
  nested. Keep the comparison fair.
- **The two effects must dissociate.** Leakage lives in `SelectKBest` being fitted
  on all data (preprocessing); selection bias lives in reporting the best-of-grid
  flat-CV mean. `pipeline` fixes only the first, `nested` fixes both — so
  `optimism(leaky) > optimism(pipeline) > optimism(nested) ≈ 0`. Don't blur them.
- **`pipeline` and `nested` ship the *same* model** (the config a flat honest
  selection on all data picks), so they share an oracle score and their reported-
  score gap isolates selection bias alone.
- **The oracle is never shown to the forge** during selection — it only scores the
  final shipped pipeline. It is the ground truth, large (`N_ORACLE`) for low
  variance.
- **Determinism**: seed every split/dataset from `Settings.rng(SALT, seed, …)`,
  never `hash()` (per-process randomised → non-reproducible benchmark).
- **Estimators stay honest, not strawmen.** The task's Bayes ceiling is < 1.0 by
  design (logistic label noise); don't crank `signal` to make a number prettier.
- The gate enforces the **shape** (both effects present and ordered, control
  agreeing, null collapsing), not just thresholds.

## Env notes
Windows console is cp1252 — the CLI prints ASCII only; the harness writes UTF-8 to
`RESULTS.md`. `gh` CLI authed as `ranafaraz`. On Windows the local default branch
is `master`; `git branch -M main` after the first push.
