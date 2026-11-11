---
type: task
title: Derive coefficient statistics and p-values from one canonical null
description: Make each tested coefficient row report the signed root of the same native Wald test whose p-value it shows, with the null canonicalised once in the fit's precision.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/coefficient-tests/README.md
status: done
---

# Derive coefficient statistics and p-values from one canonical null

## Goal

Review finding R1, accepted as row F01. `InferenceResult._coefficient_test` in
`src/python/packages/statadic/src/statadic/api/inference.py` gets its p-value from the
native test of `jnp.asarray([null], dtype=parameters.dtype)`. It computes the
displayed statistic on the host as `(float(estimate) - null) / standard_error`,
using the Python null. In float32 these are different tests: an estimate of
10000.0, a null of 10000.0003 and an SE of 1e-4 show z ≈ −3 with p = 1.

Rejecting non-representable nulls is declined, because decimal nulls such as 0.1
are almost never exactly representable. Promoting to float64 is declined, because
the precision policy forbids pinning casts. Instead, the null is canonicalised once
in the fit's dtype, and both columns come from that one native test.

## Acceptance

- [x] The signed statistic is the signed root of the native Wald quadratic, and its p-value is the native p-value of the same call.
- [x] Metadata `effective_null` and one warning appear exactly when a tested null changes under the dtype conversion; neither appears in float64.
- [x] Non-representable nulls near an estimate, positive and negative deviations, float32 and float64, normal and t references, and a small positive variance are tested.
- [x] Reports without a null still carry no statistic or p-value.
- [x] `tests/api/test_inference.py` and the statsmodels coefficient reference pass; lint passes for the touched files.

## Specification

- Files: `src/statadic/api/inference.py` (`_coefficient_test`, `report`) and
  `tests/api/test_inference.py` only.
- Canonical null: `target = jnp.asarray([null], dtype=fit.parameters.dtype)` and
  `effective = float(target[0])`. Pass `target` to `linear_restriction_test`.
- Statistic: `math.copysign(math.sqrt(wald), float(estimate) - effective)`.
  - `wald` is the unscaled single-restriction Wald quadratic: the value
    `InferenceResult.test` exposes as `details["wald_statistic"]`.
  - Read `linear_restriction_test` to find that field on the native result, and
    confirm that the F reference with one restriction does not rescale it.
  - `0.0` when `wald == 0`.
  - The p-value stays `native.test.pvalue`.
- Metadata `null` stays as given (a float or a mapping).
- When `effective != null` for any tested row:
  - Add metadata `effective_null` with the same shape: a float for a float null, or
    a mapping of the tested names otherwise.
  - Add one warning per affected row:
    `Null <requested> for <name> is <effective> in the fit's <dtype> precision; the test uses <effective>.`
    Use `repr`-style floats and the dtype name, e.g. `float32`.
- Tests:
  - Extend `test_report_pvalue_equals_single_restriction_test`, or add a sibling
    beside it.
  - Build a float32 fit by passing float32 arrays, so that ordinary promotion keeps
    float32. Do not change process precision.
  - Assert:
    - `statistic**2 ≈ details["wald_statistic"]`;
    - the p-value equals `test()` at the same requested null;
    - for the normal reference, the p-value is ≈ `2 * (1 - Φ(|statistic|))`;
    - metadata and warnings behave as above.
  - Keep the existing statsmodels walkthrough passing:
    `tests/integration/api/test_reference_walkthroughs.py`, run on the external lane.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
