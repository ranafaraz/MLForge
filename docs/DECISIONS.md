# Design decisions / what I built

A short "why it is this way" log for MLForge.

## 1. The product is a *measurement*, not a model
Plenty of repos fit a classifier. MLForge's contribution is quantifying the gap
between the score a selection procedure **reports** and the score the shipped
model **actually achieves** — and decomposing that gap into two named,
independently-corrected effects (preprocessing leakage, selection bias). Pinning
each effect to a single structural change is what makes the claims falsifiable.

## 2. Ground truth via a large oracle split
Every dataset is synthetic with a known logistic data-generating process, so we
also know the Bayes-optimal accuracy. The shipped model is scored on a large
held-out **oracle** drawn from the same process; oracle accuracy is a low-variance
stand-in for true generalization. No downloads, no labels to trust — the truth is
constructed.

## 3. From scratch on numpy, not scikit-learn
The estimators, scaler, feature selector, pipeline and cross-validation are
hand-rolled. Two reasons: (a) it keeps the offline core to a single dependency and
fully deterministic; (b) the leakage story is more convincing when you can see
exactly where the feature selector is fit relative to the fold boundary. An
optional `[sklearn]` backend exists purely to cross-check that the numbers agree
with the reference implementations.

## 4. Three protocols, one independent variable each
`leaky` → `pipeline` flips *only* where preprocessing is fit (all-data vs
in-fold). `pipeline` → `nested` flips *only* whether selection is nested. Crucially
`pipeline` and `nested` **ship the same model**, so their oracle scores are equal
and their reported-score difference isolates selection bias with nothing else
moving.

## 5. Regimes chosen to make effects appear and disappear on cue
- `highdim` (many features, few informative, a wide search grid): both effects
  bite — feature selection has a large surface to overfit, and best-of-grid has
  many noisy candidates to cherry-pick.
- `lowdim` (few all-informative features): the **control**. Nothing to leak or
  cherry-pick, so all three protocols agree. This proves the effects are caused by
  the high-dimensional search, not by the protocols themselves.
- `null` (labels independent of features): the leaky protocol still reports
  accuracy well above the 0.5 truth — a signal it *manufactured* by selecting
  features on the held-out rows. Nested CV refuses to report it. This is the
  classic "how to fool yourself" result (Ambroise & McLachlan 2002).

## 6. Small training splits are deliberate
Leakage and selection bias are strongest when data is scarce relative to the
search space — exactly the regime where practitioners are most tempted to reuse
all their data. The training splits are small on purpose; the oracle is large so
the *truth* stays sharp even though the *estimate* is hard.

## 7. The gate checks shape, not just height
`evals/gate.py` asserts the *ordering and dissociation* (leaky more optimistic
than pipeline more optimistic than nested; nested within tolerance of the oracle;
the control agreeing; the null collapsing), with floors set well below the
observed numbers so ordinary seed variation never flakes CI while a real
regression — a leak that creeps back in, a verifier of honesty that breaks —
trips it.

## Cross-checking with scikit-learn
`pip install -e ".[sklearn]"` and set `MLFORGE_BACKEND=sklearn` to re-run the same
three protocols with `sklearn.pipeline.Pipeline`, `cross_val_score` and
`GridSearchCV`; the optimism decomposition matches the from-scratch numbers.
