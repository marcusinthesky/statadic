---
type: task
title: Document fit statistics and their conventions
description: Document the fit statistics, their parameter-count convention and citations in the reporting guide, reference page, OLS example and changelog.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: done
---

# Document fit statistics and their conventions

## Goal

Start when `table-content`, `fit-labels`, `likelihood-statistics` and
`spatial-statistics` are done. Researchers must see which R² they get and how the
criteria count parameters, because these conventions differ across packages.

## Acceptance

- [x] The reporting guide shows a regression table with N, R² and the criteria, and says which R² is reported, how `k` is counted, and the offset from statsmodels for OLS.
- [x] It cites Akaike (1974), Schwarz (1978) and Hannan and Quinn (1979), and says criteria compare models of the same response on the same rows only.
- [x] The reference page lists the metric names, `FitStatistics`, `fit_statistics()` and the families that report each statistic.
- [x] `CHANGELOG.md` records the fit statistics without compatibility framing.
- [x] The docs size budget, example checks and docs reporting tests pass, and `rumdl check` passes on the edited pages.

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

- [Remove transitional fit imports and document the contracts](../../../final-review/tasks/fit-cleanup-docs/README.md)

<!-- insitu:end -->
