---
type: document
title: Directional coefficient test design
description: Proposal for a one-sided test of one linear restriction, with the reserved-slot alternative.
links:
  - .context/plan/api-cohesion/tasks/followups/tasks/directional-tests/README.md
---

# Directional coefficient test design

Read against `00e7d8f`. This is a proposal for the owner decision in the
[task](README.md); nothing here is implemented or approved.

## What exists

- `Hypothesis.alternative` accepts any `Proposition`, but
  `InferenceResult._compile_hypothesis` (`api/inference.py`) raises
  `ValueError("Coefficient tests support the complement alternative only.")`
  for any non-`None` value. `api/dyadic.py`, `api/association.py` and
  `api/comparison/_results.py` do the same with their own messages.
- `InferenceResult.test(hypothesis, *, reference: "chi2" | "f",
  degrees_of_freedom=None)` calls `linear_restriction_test` in
  `statadic.inference.parameters`. That kernel computes
  `delta = R beta - r` and `q = delta' (R V R')^-1 delta` through
  `_restriction_quadratic`, which requires full row rank and a positive-definite
  restricted covariance. It returns `q` (`wald_statistic`) and a chi-square or F
  tail. The sign of `delta` is discarded.
- `coefficient_intervals` already offers the `"normal"` and `"t"` references,
  with explicit `degrees_of_freedom` for `"t"`.
  `statadic.inference._parameter_reference` holds `_student_two_sided_pvalue`
  (regularized incomplete beta, validated to `_MAX_VALIDATED_DEGREES = 10000`)
  and `_outside_validated_range`.
- `Normal(location, scale)` is a `Reference`; `ResidualMoran`
  (`api/spatial/diagnostics.py`) is the only producer and records
  `construction["tail"] = "two-sided"`. `ResampledReference` is the only
  reference with a `tail` field. There is no Student-t reference value.

For one restriction row, chi2(1) is the square of a standard normal and F(1, d)
is the square of Student t(d), so the existing two-sided p-values are exactly
twice the one-sided tail in the direction of the estimate.

## Option A: support a directional alternative (recommended)

### Proposition and spelling

A new immutable proposition states only the direction. The restriction stays in
the null, so the row, target and design binding cannot disagree between null and
alternative.

```python
from statadic.api.hypotheses import Hypothesis, LinearRestriction, Greater

claim = Hypothesis(
    LinearRestriction({"x": 1.0}, values=0.0),   # or LinearRestriction.zero("x")
    alternative=Greater(),
)
outcome = inference.test(claim, reference="t", degrees_of_freedom=fit.df_resid)
outcome.pvalue            # P(T >= observed t) under the boundary null
```

`Greater()` and `Less()` are two frozen dataclasses (or one
`Directional(side: Literal["greater", "less"])` with both spellings as
factories; the owner chooses, behavior is identical). Each implements the
`Proposition` protocol:

- `describe()`: "The restricted combination exceeds its null target." or
  "... is below its null target."
- `metadata()`: `{"kind": "directional", "side": "greater"}`.

They are `__all__` exports of `api/hypotheses.py`. `Hypothesis` already prints
`"...; alternative: ..."` and snapshots both metadata mappings, so no change to
the wrapper is needed beyond updating its docstring (it currently says
first-slice procedures require the default).

The null stays the equality `R theta = r`. The stated question is the
boundary-null one-sided problem `H0: R theta = r` against `H1: R theta > r`
(or `<`); reports must not describe the null as the inequality.

### Statistic and reference

With one row, `v = R V R'` (scalar), `delta = R beta - r`:

- `t = delta / sqrt(v)`, signed, so `t = sign(delta) * sqrt(q)`.
- Reference `"normal"`: standard normal, no `degrees_of_freedom` (same rule as
  `coefficient_intervals`). `p_greater = Phi(-t)`, `p_less = Phi(t)`.
- Reference `"t"`: Student t with explicit finite positive
  `degrees_of_freedom`. With `p2 = _student_two_sided_pvalue(t, df)`, the tail in
  the direction of the sign is `p2 / 2`; the opposite tail is `1 - p2 / 2`.
- `delta == 0` gives `t = 0` and `p = 0.5` in both tails.
- As today, robust covariance alone does not justify an exact t law; the
  reference remains the caller's stated assumption and the existing assumptions
  string is retained.

Reuse, by name:

| Piece | Source | Use |
| --- | --- | --- |
| `_restriction_arrays`, `_reference_degrees` | `inference/parameters.py` | shape, name-order and df validation, unchanged |
| `_restriction_quadratic` (and its `_restricted_covariance`) | `inference/parameters.py` | rank, finiteness, positive-definite checks; `v > 0` is the same condition as the 1x1 solve |
| `_availability`, `InferenceStatus` | `inference/parameters.py`, `core.results` | status preservation, `INVALID_COVARIANCE`, `INVALID_RESTRICTION`, `INVALID_REFERENCE` |
| `_student_two_sided_pvalue`, `_outside_validated_range` | `inference/_parameter_reference.py` | t tail and validated-range flag |
| `jax.scipy.stats.norm` | JAX | normal tails (`normal_critical_value` already uses `ndtri`) |
| `Normal` | `api/hypotheses.py` | normal reference as `Normal(0, 1)` |

New pieces:

1. A native kernel `directional_restriction_test(uncertainty, restriction,
   value, *, side, reference, parameter_names=None, degrees_of_freedom=None)` in
   `inference/parameters.py`, beside `linear_restriction_test`, returning a
   frozen pytree result with `test: TestResult` (signed statistic and one-sided
   p-value), `estimate_delta`, `standard_error`, `side`, `reference`,
   `valid`, `status` and the finite-range flag. It needs `delta`; the cleanest
   route is to have `_restriction_quadratic` also return it (it computes it
   already) rather than recompute it. Batch behavior follows the existing
   kernel (per-lane NaN plus status under eager, jit and vmap). `side` and
   `reference` are static metadata fields like `LinearRestrictionResult.reference`.
2. A `StudentT(df, *, tail)` `Reference` value in `api/hypotheses.py`, joining
   the `Reference` union and the `isinstance` check in `TestOutcome`.
3. A `tail: Literal["two-sided", "greater", "less"]` keyword on `Normal` and
   `StudentT`, default `"two-sided"`, mirroring `ResampledReference.tail`. The
   default keeps `ResidualMoran` unchanged. `__str__` includes a non-default
   tail, for example `t(12), upper tail`, so `"Calibrated p-value under {reference}"`
   in `TestOutcome.metrics` names the tail.

`FisherF` and `ChiSquare` gain no tail; they remain upper-tail squares of a
two-sided law.

### `InferenceResult.test` signature

```python
def test(
    self,
    hypothesis: Hypothesis[LinearRestriction],
    *,
    reference: Literal["chi2", "f", "normal", "t"],
    degrees_of_freedom: ArrayLike | None = None,
) -> TestOutcome[LinearRestriction]:
```

Dispatch is on `hypothesis.alternative` and `reference` together:

- `alternative is None` with `"chi2"` or `"f"`: unchanged.
- `alternative` is `Greater`/`Less` with `"normal"` or `"t"`: directional path.
- All other combinations raise (below). In particular `"normal"` or `"t"` without
  a stated direction is rejected, not silently made two-sided; a two-sided signed
  test adds nothing over `"f"`/`"chi2"` for one row and would add a second
  spelling.

### What stays rejected, and exactly how

All are eager `ValueError`/`TypeError` raised by the host before any
calculation, matching the existing convention that shape and spelling errors
raise while numeric failures stay as unavailable outcomes.

| Case | Behavior |
| --- | --- |
| Directional alternative with more than one restriction row | `ValueError("A directional alternative requires exactly one restriction row; joint restrictions have no one-sided Wald law.")` |
| Directional alternative with `"chi2"` or `"f"` | `ValueError("A directional alternative requires reference 'normal' or 't'.")` |
| `"normal"`/`"t"` with `alternative=None` | `ValueError("References 'normal' and 't' require a directional alternative.")` |
| `"t"` without `degrees_of_freedom`; `"normal"` with it | existing `_reference_degrees` `ValueError`s |
| Any alternative that is not `Greater`/`Less` (an arbitrary `Proposition`) | `ValueError("Coefficient tests support the complement or a directional alternative only.")`; this replaces the current message |
| Null that is not a `LinearRestriction` | existing `TypeError` |
| Directional alternative on dyadic, association, PERMDISP or Mantel tests | unchanged messages; their laws are resampled or tail-fixed and are out of scope |
| Boundary or chi-bar laws, inequality-constrained joint alternatives | stay deferred under A20 in the [dispositions](../../../../proposal-dispositions.md); no proposition is introduced for them, and the `Directional` proposition does not accept them |
| Resampled one-sided families | out of scope; `ResampledReference.tail` is untouched |

Numerical unavailability is not an error. A zero or non-finite `v`, a singular
restricted covariance, an invalid row or an invalid df gives
`available=False` with `unavailable_reason` from `InferenceStatus`
(`INVALID_COVARIANCE`, `INVALID_RESTRICTION`, `INVALID_REFERENCE`, upstream
status preserved), `statistic=None`, `pvalue=None` and `reference=None`,
exactly as the Wald path does today. A one-sided p-value is never produced from
an unavailable covariance, and `delta`'s sign alone is never reported as a
result.

### Presentation

`TestOutcome` keeps its fields; the following are populated.

- `name`: `"directional-wald"`, to keep it distinct from `"wald"` in metrics keys
  (`directional-wald.statistic`, `.pvalue`).
- `statistic`: the signed t (or z). `reference`: `Normal(0, 1, tail="greater")`
  or `StudentT(df, tail="greater")`; `pvalue`: that tail's probability.
- `details`: `estimate_delta`, `standard_error`, `restriction_rank`,
  `covariance_rank`, `denominator_df` (t only). `wald_statistic` is not shown,
  since it equals `t**2` and would invite reading it as the statistic.
- `construction`: the existing keys plus `"tail"` (`"greater"`/`"less"`, the same
  key `ResidualMoran` already writes), `"requested_reference"`, and the
  finite-range flag.
- `Report.metadata["reference"]` is `reference.metadata()`, so it carries
  `{"kind": "student-t", "df": ..., "tail": ...}`. `Report.metadata["hypothesis"]`
  already carries the alternative metadata, so the stated direction and the
  tail agree in two places; a construction check asserts they do.
- `TestOutcome.report()` warnings and the finite-range note are unchanged. One
  extra note: "One-sided p-value for a boundary null; the opposite tail is
  1 - p." This sits beside the existing failure-to-reject note.

`ChiSquare`/`FisherF` serialization is unchanged. Existing `Normal` metadata
gains `"tail": "two-sided"`; the owner should accept that the Moran report
metadata then gains one key.

## Option B: document the slot as reserved

Change no behavior. Edit the `Hypothesis` docstring and the four rejection
messages to say the slot is reserved: for example "Coefficient tests support
the complement alternative only; directional alternatives are not implemented."
Add one contract test that every `.test` rejects a non-`None` alternative.
Cost: near zero and no new scientific claims. Drawback: a public constructor
argument that no procedure honors, and nothing the user can write to express
the commonest applied question (sign of one coefficient or contrast).

## Recommendation

Option A, scoped exactly as above, when the owner accepts it. The statistic is
already computed inside `linear_restriction_test`; the only new numerics are a
sign and a halved existing tail. The reference value, the tail field and the
proposition are small, typed and follow existing precedent
(`ResampledReference.tail`, `construction["tail"]`). It converts the slot from a
trap into a supported statement while keeping everything that needs real new
theory (joint, boundary, chi-bar, resampled) rejected. Until the owner decides
and a workflow triggers it, Option B's wording changes are the safe interim and
are compatible with A, since A only narrows the rejection.

## Verification plan

All expected values come from independent code, never from the kernel under
test. statsmodels and scipy are declared dependencies of the package.

1. One-sided t against statsmodels. Fit OLS on a fixed dataset; compare
   `results.t_test("x = c", alternative="larger" / "smaller")` (statistic and
   p-value) and `scipy.stats.t.sf`/`cdf` computed from the independently
   obtained `beta` and covariance with `reference="t"`,
   `degrees_of_freedom=df_resid`. Include a nonzero target, a two-coefficient
   contrast row, and a negative `delta` so the tail swap is exercised.
2. Normal reference with robust covariance. Use statsmodels
   `cov_type="HC1", use_t=False` (and a clustered case) for the covariance and
   `scipy.stats.norm.sf` for the tail; compare against `reference="normal"`.
3. Algebraic identities as property checks: `p_greater + p_less = 1`;
   `2 * min(p_greater, p_less)` equals the existing `"f"` p-value (df
   `(1, d)`) and the `"chi2"` p-value for the same row; `statistic**2` equals
   `wald_statistic`; `delta == 0` gives `p = 0.5`.
4. Far tail and large df: a statistic near 30 and `df` at and above the
   validated limit of 10000 against `scipy.stats.t.sf`; the finite-range flag is
   set above the limit; `t` with very large df approaches the normal result.
5. Unavailable covariance. Build fits whose uncertainty is invalid (rank
   deficient design, exact fit, `valid=False` uncertainty) and a row with
   `v = 0` (contrast in the null space of the covariance). The oracle here is the
   existing Wald path on the same inputs: both must report `available=False` with
   the same `InferenceStatus` name, and the directional outcome must have
   `statistic=None`, `pvalue=None`, `reference=None`. Assert that no p-value is
   computed from the sign of `delta` alone.
6. Rejection matrix: one test per row of the table above, asserting exception
   type and message, plus a check that `alternative=None` with `"chi2"`/`"f"` is
   byte-identical to current output.
7. Native kernel under `jit` and `vmap` over batched fits, with one invalid lane
   producing NaN and a status while the others stay valid, mirroring the
   existing restriction-kernel tests.
8. Report checks: `reference.metadata()["tail"]` equals
   `hypothesis.metadata()["alternative"]["side"]`; `construction["tail"]`
   matches; the p-value metric text names the tail.

Per the repository's reference workflow, add the statsmodels/scipy comparison
as a catalogue entry in the existing reference runner with exact source, lock
and fixture identities. That entry and any catalogue edit belong to the
implementing task, not this design.
