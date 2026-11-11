---
type: task
title: Document coefficient tests, test tables and value export
description: Document coefficient tests, test tables and value export in the reporting guide and reference, show them in the OLS example, and record them in the changelog.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/coefficient-tests/README.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/test-tables/specification.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/inline-values/specification.md
status: done
---

# Document coefficient tests, test tables and value export

## Goal

Start when `coefficient-tests`, `test-producers` and `inline-values` are done.
Researchers find these features in the reporting guide and reference page, and the
OLS example shows them in use.

Source under `src/python/packages/statadic/src/` and tests are not edited here.

## Acceptance

- [x] The reporting guide shows `report(reference=..., null=0)`, a table of diagnostics, and `save_values` with `\input`, `\jcnum` and `$p \jcvalue{...}$`.
- [x] The guide says a coefficient test is opt-in, two-sided and equal to `InferenceResult.test` for one restriction.
- [x] The reference page lists `TestRow`, `Report.tests`, `null=`, `to_values` and `save_values`, the key scheme, and that unavailable values are not declared.
- [x] `examples/ols_analysis.py` saves a coefficient table with tests, a diagnostics table and a values file, and the docs example checks pass.
- [x] `CHANGELOG.md` records the three features, the removed `TestOutcome.metrics()` and the artifact schema version 7, with no compatibility framing.
- [x] The docs size budget, example checks and docs reporting tests pass, and `rumdl check` passes on the edited pages.

## Specification

- Files: `src/python/packages/statadic/docs/guides/reporting.md`,
  `src/python/packages/statadic/docs/reference/reporting.md`, any page naming
  `TestOutcome.metrics()`, `src/python/packages/statadic/examples/ols_analysis.py` and
  `src/python/packages/statadic/CHANGELOG.md`.
- Keep pages short; the docs app enforces a size budget.
- State the measured width limit. A coefficient table with tests and intervals has
  seven columns and overflows a default single-column `article` by about 29pt at
  four digits. Inside the float, `\small` with `\setlength{\tabcolsep}{4pt}` fits;
  so does a lower `digits`. Layout stays with the manuscript.
- Say what the features do not establish: a p-value states the stated null under
  the stated reference law; it does not establish the reference law, the model or
  coverage.

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
