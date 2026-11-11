---
type: task
title: Make Report.to_latex fit a portrait article
description: "Make the report's LaTeX table fit a portrait page, omit empty columns, state its interval level and keep metadata out of the body."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Make Report.to_latex fit a portrait article

## Goal

`Report.to_latex()` must produce a table that fits a portrait `article` and states what its intervals mean, while retained report values stay unchanged.

## Acceptance

- [x] The evidence fixture's `fit.report()` and `report` compile in a plain `article` with `booktabs` and no Overfull box.
- [x] Columns other than Parameter and Estimate are omitted when every row's value is `None`. Retained `ParameterRow` values do not change.
- [x] An invalid row still shows `unavailable` and keeps its reason as a visible note.
- [x] Numbers are typeset in math mode, with exponents as `\times 10^{n}`. A `digits` keyword defaults to 6.
- [x] Generic metadata moves to `%` comment lines, so no JSON paragraph prints.
- [x] `InferenceResult.report` retains `confidence` in metadata and adds a visible note such as `95% intervals; normal reference; covariance HC1`.
- [x] A new `Report.notes` field renders in text, HTML and LaTeX and round-trips through analysis snapshots.
- [x] `apps/docs/templates/report.tex` drops landscape and `\footnotesize`, and the Overfull rejection stays.
- [x] `just python::docs::check` and `just check` pass.

## Specification

- Main code: `to_latex` and `_display` in `api/reporting.py`, `report` in `api/inference.py`, `_report_state` and `_restore_report` in `api/artifacts/_snapshot.py`.
- Adding `notes` changes persisted content, so apply the schema rule from the handoff guide.
- Separate consecutive tabulars with `\par\medskip`.
- Keep the first line as the `%` title comment. `to_text` and `to_html` keep their columns.

Existing tests constrain the output and should keep passing:

- `tests/api/test_reporting.py:93` keeps reasons and escaped labels in LaTeX.
- `apps/docs/tests/test_reporting_workflow.py` expects `format(value, ".6g")` values, `"HC3"`, and a first line of `% OLS inference`.

Add a compile test for the fixture to `test_reporting_workflow.py`.

## Non-goals

No caption, label or float wrapper, because the manuscript owns those. No multi-model tables, rate ratios or third-party table libraries.

Follow the [handoff guide](../../handoff.md). Measurements are in [evidence](../../evidence.md).

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

- [Decline third-party LaTeX table libraries](../latex-table-libraries/README.md)
- [Add coefficient-interval and prediction-line helpers](../plot-helpers/README.md)
- [Defer StatisticalTable and rate-ratio tables](../statistical-table/README.md)

<!-- insitu:end -->
