---
type: task
title: One estimation method and explicit OLS diagnostics
description: Rename fit to estimate everywhere and add result.diagnose with qualified OLS residual tests.
links: []
status: done
---

# One estimation method and explicit OLS diagnostics

## Goal

Give STATADIC one estimation method, `estimate`, so no `fit` suggests compatibility
with scikit-learn's protocol. Add one explicit diagnostic entry point,
`result.diagnose(*requests)`, with OLS residual and influence diagnostics that
carry reference parity, a size study, Lean claims and citations.

## Acceptance

- [x] `Model.estimate`, `estimate_prepared` and `Estimator.estimate` replace
  `fit`; a contract test rejects any public `fit` method.
- [x] Registry symbols, public API inventory, docs and examples use `estimate`.
- [x] `LinearResult.diagnose` accepts only typed requests and keeps unavailable
  outcomes with reasons; diagnostics never refit or change the estimate.
- [x] Jarque-Bera, Breusch-Pagan (studentized and original), White,
  Durbin-Watson, Breusch-Godfrey, RESET and influence match statsmodels and
  independent NumPy or leave-one-out witnesses in float32 and float64.
- [x] A prespecified 10,000-trial Gaussian-null study qualifies the size of
  every calibrated test.
- [x] Lean claims STATADIC-DW-001, STATADIC-JB-001 and STATADIC-INFLUENCE-001 compile and
  are registered with sources.
- [x] `just check` passes before handoff (2026-11-08).

## Notes

The asymptotic chi-square calibration remains open as
`linear-diagnostics-002`. Follow-ups are
[family diagnostics](../family-diagnostics/README.md) and
[prediction options](../prediction-options/README.md).

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
