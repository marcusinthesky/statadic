---
type: task
title: Compare HQIC and BIC with peer formulas
description: Check HQIC, and BIC where missing, for likelihood regression, hurdle and survival fits against statsmodels eval_measures on the peer log likelihood.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/likelihood-statistics/README.md
status: done
---

# Compare HQIC and BIC with peer formulas

## Goal

Only OLS and WLS compare HQIC with a peer. statsmodels GLM, discrete models and
lifelines report no HQIC, and the hurdle's BIC is checked only by formula.
`statsmodels.tools.eval_measures` implements `aic`, `bic` and `hqic` from a log
likelihood, `nobs` and a parameter count. Applying it to the peer's matched log
likelihood with statadic's `k` and `n` checks statadic's criteria against an independent
implementation (row F10).

## Acceptance

- [x] Every existing external peer test of the Poisson and grouped-logit GLM, gamma, NB2 (estimated and fixed), probit, hurdle, AFT and spatial-survival fits also asserts HQIC through `eval_measures.hqic`.
- [x] The hurdle asserts BIC through `eval_measures.bic` with the summed peer log likelihood.
- [x] Each docstring states `k`, `n` and the peer convention; existing tolerances are kept.
- [x] Modules that newly use statsmodels carry `requires_optional("statsmodels")`.
- [x] The touched tests pass on the external lane with no skips; lint passes.

## Specification

- Files: `tests/api/regression/test__glm.py`,
  `tests/api/regression/test__likelihood.py`,
  `tests/api/regression/test__conditional.py`,
  `tests/testing/test_survival_references.py` and
  `tests/api/survival/test__spatial.py`.
- Read `statsmodels/tools/eval_measures.py` in the reference environment
  (`.cache/statadic-reference-venv`) and confirm `hqic(llf, nobs, df_modelwc)` is
  `-2 llf + 2 log(log(nobs)) df_modelwc`.
- Use the peer's own log likelihood. Where a test already reconstructs it (the
  gamma profile, the hurdle sum, log time minus `sum(log t)`), use that.
- Use statadic's `n_parameters` and the `n` its docstring states.
- `test_survival_references.py` is lifelines-marked at module level. Add
  `requires_optional("statsmodels")` to the tests that use statsmodels.

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
