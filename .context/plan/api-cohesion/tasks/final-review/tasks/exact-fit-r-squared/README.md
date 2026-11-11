---
type: task
title: Report R-squared for valid exact least-squares fits
description: Separate the availability of R-squared, adjusted R-squared and the Gaussian likelihood so that a valid exact fit reports R-squared of one.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: done
---

# Report R-squared for valid exact least-squares fits

## Goal

Review finding R2, accepted as row F02. In
`src/python/packages/statadic/src/statadic/api/base.py`, `_linear_fit_statistics` gates both
the Gaussian likelihood and R² on `usable = valid and n > 0 and weighted_sse > 0`.
A valid exact fit (SSE = 0) with positive total variation therefore loses its
R² = 1. `LinearFit.valid` is true at SSE = 0, because its common scale is a finite
zero (`estimation/linear.py`). One quantity's failure must not erase another.

## Acceptance

- [x] Each quantity has its own availability predicate and reason, as specified.
- [x] Exact fits report R² = 1, while their likelihood and criteria stay unavailable and `n_parameters` stays available.
- [x] Constant-response and saturated-fit controls report the specified unavailability.
- [x] `report()` and `infer(...).fit_statistics` carry the same metrics.
- [x] `tests/api/regression/test__linear.py` and the statsmodels fit-statistics reference pass; lint passes for the touched files.

## Specification

- Files: `src/statadic/api/base.py` (`_linear_fit_statistics`, `_r_metric`) and
  `tests/api/regression/test__linear.py` only.
- `point_valid = bool(core.valid) and n > 0`.
- Likelihood and criteria: available when `point_valid and weighted_sse > 0`. The
  reason is unchanged.
- Centred:
  - Available when `point_valid and total > 0`, where
    `total = Σ w (y - ȳ_w)²`. The value is `1 - SSE_w / total`, so 1 at an exact
    fit.
  - Reasons: `Needs a valid least-squares fit.` or
    `The response has no variation about its weighted mean.`
- Adjusted: available when `r_squared` is available and `n > rank`. Otherwise use
  the R² reason, or `Needs positive residual degrees of freedom (n > rank).`
- Uncentred: available when `point_valid and Σ w y² > 0`. Reasons:
  `Needs a valid least-squares fit.` or `The response is zero in every row.`
- `_r_metric` takes its reason explicitly. Notes, metric names and order are
  unchanged.
- Tests:
  - OLS exact fit with an intercept.
  - Uncentred `x=(0,1,0)`, `y=(0,2,0)`: `uncentred_r_squared == 1`.
  - WLS exact fit.
  - Constant response with an intercept: R² is unavailable with the variation
    reason.
  - Saturated `n == rank`: R² = 1, adjusted unavailable.
  - In each case, check report and inference propagation.
- The external lane still passes
  `tests/integration/api/test_reference_walkthroughs.py -k fit_statistics`.

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
