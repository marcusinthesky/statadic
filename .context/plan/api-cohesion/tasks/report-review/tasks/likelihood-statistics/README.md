---
type: task
title: Report information criteria for likelihood regression and survival fits
description: Report n_parameters, log likelihood, AIC, BIC and HQIC for GLM, generalised-likelihood, hurdle and survival fits, matched to installed peers.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: done
---

# Report information criteria for likelihood regression and survival fits

## Goal

Extend the fit statistics of [table-content](../table-content/README.md) to every
non-spatial family with a retained log likelihood. Start when `table-content` is
done. The [specification](../table-content/specification.md) is normative.

## Acceptance

- [x] GLM, generalised-likelihood (gamma, NB2, probit), hurdle and survival AFT results override `fit_statistics()` with the shared helper, and their `infer` passes it on.
- [x] Each family's `k` follows the counting rule; the test file states it per family, including fixed versus estimated dispersion and the hurdle's two parts.
- [x] External-reference tests match the log likelihood of statsmodels GLM and discrete models and lifelines AFT fitters, and the criteria through each peer's stated offset (statsmodels GLM: use `bic_llf`, not the deviance BIC).
- [x] No R² is added to these families.
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

- [Compare HQIC and BIC with peer formulas](../../../final-review/tasks/criterion-peers/README.md)

<!-- insitu:end -->
