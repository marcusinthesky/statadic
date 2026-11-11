---
type: task
title: Report fit statistics in fit and inference reports
description: Add the shared fit-statistics record and helper, report R², log likelihood, AIC, BIC and HQIC for OLS and WLS, and carry fit statistics into inference reports.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
  - .context/reference/akaike-1974/README.md
  - .context/reference/schwarz-1978/README.md
  - .context/reference/hannan-quinn-1979/README.md
status: done
---

# Report fit statistics in fit and inference reports

## Goal

The owner decided on 2026-11-11 (row P08 of the
[dispositions](../../../../proposal-dispositions.md)): R² is required and must say
whether it is adjusted, and the log likelihood, AIC, BIC and HQIC should be shown
with their original citations. This task builds the shared pieces and the linear
family; [likelihood statistics](../likelihood-statistics/README.md) and
[spatial statistics](../spatial-statistics/README.md) extend them. The
[specification](specification.md) is normative.

## Acceptance

- [x] `FitStatistics`, `Result.fit_statistics()` and `information_metrics` exist in `api/base.py` as specified.
- [x] `InferenceResult` carries `fit_statistics` and its report appends the metrics after `n_observations` and the notes after its own.
- [x] OLS and WLS fit reports retain `n_parameters`, the R² metrics, `log_likelihood`, `aic`, `bic` and `hqic`, with the specified notes; `infer` passes them on.
- [x] Designs with an explicit intercept, an implicit constant (fixed-effect indicators) and no constant each choose the specified R².
- [x] An external-reference test matches statsmodels OLS and WLS, with and without a constant: `llf`, `rsquared`, `rsquared_adj` exactly, and AIC, BIC and HQIC after the stated offset for statsmodels' parameter count.
- [x] The named test files for the touched modules pass; lint and strict typing pass for the touched files.

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

- [Add opt-in coefficient tests to inference reports](../coefficient-tests/README.md)

<!-- insitu:end -->
