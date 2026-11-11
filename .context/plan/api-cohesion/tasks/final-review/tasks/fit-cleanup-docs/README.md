---
type: task
title: Remove transitional fit imports and document the contracts
description: Delete the transitional fit-statistics re-exports and document the fit-statistics owner, availability rules, footer selection and estimand naming.
links:
  - .context/plan/api-cohesion/tasks/final-review/tasks/fit-contract/README.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/fit-docs/README.md
status: done
---

# Remove transitional fit imports and document the contracts

## Goal

This closes the final review: no transitional names remain, and the documentation
states the contracts that every earlier task changed.

## Acceptance

- [x] `api/base.py` has no transitional re-exports, and nothing in `src`, the tests, the apps or the docs imports fit-statistics names from `api.base`.
- [x] The reporting reference documents:
  - the `statadic.api.fit_statistics` names;
  - `LikelihoodSummary` and the two Protocols;
  - the R² and likelihood availability rules;
  - the effective null in reduced precision;
  - the Normal reference formatting;
  - `to_latex(metrics=...)`.
- [x] The estimand reference separates the `Estimand` workflow from the `core.statistics.PopulationEstimand` axis, and the contrast docs mention `LinearPredictorContrast`.
- [x] `api.rst`, the docs nitpick lists, the package `ARCHITECTURE.md` and `CHANGELOG.md` reflect the changes, without compatibility framing.
- [x] `rumdl check` passes on the edited pages, and the docs reporting and example tests pass.

## Specification

- Files: `src/statadic/api/base.py` (removing the transitional block), any remaining
  importer found with
  `git grep -n "from statadic.api.base import" -- src/python | grep -E "FitStatistics|information_metrics|_INFORMATION_NOTE|_linear_fit_statistics"`,
  `docs/reference/reporting.md`, `docs/guides/reporting.md`,
  `docs/reference/api.rst`, `docs/reference/estimand.md`, the docs page that
  describes contrasts (search for `RowDifference`), `src/python/apps/docs/conf.py`,
  `src/python/packages/statadic/ARCHITECTURE.md` and `CHANGELOG.md`.
- In `src/statadic/inference/gms.py`, correct the `indicator_gms` docstring: QLR runs on raw `√T·ḡ` with `pinv(Ω)`, not on studentized moments. The two are equivalent because the orthant is invariant to positive diagonal scaling (found by `pvalue-witnesses`).
- Do not touch `src/lean`, the catalogue or the records; the coordinator owns them.

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
