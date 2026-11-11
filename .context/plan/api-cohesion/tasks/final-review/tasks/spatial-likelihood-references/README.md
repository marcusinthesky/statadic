---
type: task
title: Compare spatial panel and SAC likelihoods with references
description: Compare random-effects and pooled spatial panel likelihoods with spreg, witness the SAC likelihood independently and add HQIC to the spatial walkthroughs.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/spatial-statistics/README.md
status: done
---

# Compare spatial panel and SAC likelihoods with references

## Goal

Several spatial fits have no independent comparison of their log likelihood and
criteria (row F10):

- SAC is checked only by formula on statadic's own likelihood.
- Pooled and random-effects spatial panels are checked only by formula.
- Observed-panel and graph routes agree only with the complete-panel route.

spreg provides `Panel_RE_Lag`, `Panel_RE_Error`, `ML_Lag` and `ML_Error`. A pooled
panel with block-diagonal weights `kron(I_T, W)` reduces exactly to a
cross-section on stacked data.

## Acceptance

- [x] Random-effects SAR and SEM panels are compared with spreg `Panel_RE_Lag` and `Panel_RE_Error`, covering coefficients, log likelihood, and AIC and BIC through stated offsets. Otherwise the test records why the models differ.
- [x] Pooled SAR and SEM panels are compared with `ML_Lag` and `ML_Error` on stacked data with `kron(I_T, W)`.
- [x] SAC is witnessed by a dense numpy Gaussian log likelihood at statadic's estimate, plus a SciPy check that no nearby parameter improves it beyond tolerance.
- [x] The existing spreg walkthroughs also assert HQIC through `statsmodels.tools.eval_measures.hqic`.
- [x] The module docstring records the fits with no peer: learned graph-parameter counts and censored spatial survival with estimated `rho`.
- [x] The touched tests pass on the external lane with no skips; lint passes.

## Specification

- Files: `tests/integration/api/test_spatial_reference_walkthroughs.py` (HQIC
  additions only) and a new `tests/integration/api/test_spatial_likelihood_references.py`.
  `integration/` is exempt from the layout mirror.
- Build statadic fits through the public `Regression` API, as the existing walkthroughs
  do. After `spatial-log-likelihood-name`, the native field is
  `core.log_likelihood`.
- Before asserting anything against spreg, read its source in
  `.cache/statadic-reference-venv` to confirm the model it fits (Elhorst random
  effects), its `k` and its `logll` normalisation. State each offset in the test,
  as the existing walkthroughs do for `n/2 log n` and the scale count.
- SAC log likelihood:

  ```text
  -n/2 log(2π σ²) - e'e / (2σ²) + log|I - ρW| + log|I - λW|
  e = (I - λW)((I - ρW)y - Xβ)
  ```

  Evaluate it at statadic's estimate and profiled `σ²`, and match
  `fit_statistics()["log_likelihood"]`. Then check optimality by perturbing
  `(β, ρ, λ, σ²)` along each coordinate, or by minimising from statadic's estimate with
  SciPy, and asserting no improvement beyond a stated tolerance.
- Markers: one scope marker plus `reference`, and `external_reference` with
  `requires_optional("spreg")`, `"libpysal"` and `"statsmodels"` as used.
- Source files may not be edited. Report any mismatch instead of fixing it.

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

- [Require a score check before random-effects panels report convergence](../panel-score-convergence/README.md)

<!-- insitu:end -->
