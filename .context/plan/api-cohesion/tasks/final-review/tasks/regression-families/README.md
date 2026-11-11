---
type: task
title: Move regression families to likelihood summaries
description: Make linear, GLM, generalised-likelihood and hurdle results supply a LikelihoodSummary and import fit statistics from their owner.
links:
  - .context/plan/api-cohesion/tasks/final-review/tasks/fit-contract/README.md
status: done
---

# Move regression families to likelihood summaries

## Goal

After [fit-contract](../fit-contract/README.md), each family states its `k`, `n`
and log likelihood once in `likelihood_summary()`, and `Result` derives its fit
statistics from that.

## Acceptance

- [x] `LinearResult`, the GLM results, `LikelihoodResult` and `HurdleResult` implement `likelihood_summary()`, with `k` and `n` as their current docstrings count them.
- [x] Overrides that only wrapped `information_metrics` are deleted; `LinearResult` keeps an override for R², and `HurdleResult` implements both methods.
- [x] No module imports `_INFORMATION_NOTE`, `_linear_fit_statistics` or `information_metrics` from `api.base`.
- [x] Every metric value, name, order and note is unchanged; the family tests assert the summary (`k`, `n`) and static `SummarizesLikelihood` conformance.
- [x] The four family test files pass, along with their external references; lint passes.

## Specification

- Files: `src/statadic/api/regression/{_linear,_glm,_likelihood,_conditional}.py` and
  `tests/api/regression/{test__linear,test__glm,test__likelihood,test__conditional}.py`.
- Pattern:
  - Move the counting docstring of today's `fit_statistics` onto
    `likelihood_summary`.
  - The log likelihood is the native `float(core.log_likelihood)`, which is NaN
    when the fit is invalid.
  - The `unavailable_reason` is the family's current reason string.
- `LinearResult`:
  - `likelihood_summary()` returns `gaussian_least_squares_summary(self.core)`.
  - `fit_statistics()` returns `linear_fit_statistics(self.core, weighted=self.weighted)`.
- GLM: `k` is the rank, or the width when the rank is negative.
- `LikelihoodResult`: `k` is the parameter width.
- `HurdleResult`, which is not a `Result`:
  - The summary sums the gate and count log likelihoods and parameters, with `n`
    the gate rows.
  - `fit_statistics()` returns `FitStatistics.from_likelihood(self.likelihood_summary())`.
- Import from `statadic.api.fit_statistics`.

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
