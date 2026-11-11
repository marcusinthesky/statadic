---
type: task
title: Select fit statistics in LaTeX table footers
description: Let to_latex and save_latex choose which retained metrics appear, in which order, in a table footer.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/fit-labels/README.md
status: done
---

# Select fit statistics in LaTeX table footers

## Goal

An inference table footer shows every retained metric: N, R², the log likelihood
and three criteria. A manuscript table often wants a subset in a chosen order. The
owner accepted footer selection on 2026-11-11. It is a rendering choice, so the
retained report is unchanged.

## Acceptance

- [x] `to_latex` and `save_latex` take `metrics: Sequence[str] | None = None`. `None` renders today's output byte-for-byte.
- [x] A sequence renders exactly those metrics, in that order, in the parameter-table footer and in the metric-only table; `()` renders no footer.
- [x] An unknown or duplicated name raises `ValueError` naming it.
- [x] Notes, `to_text`, `to_html` and `to_values` are unchanged, and the docstring points to `notes=False` for omitting notes.
- [x] The docs compile test renders a selected footer; the named tests pass; lint passes.

## Specification

- Files: `src/statadic/api/reporting.py`, `tests/api/test_reporting.py` and
  `src/python/apps/docs/tests/test_reporting_workflow.py`.
- Filter `report.metrics` once in `to_latex` and pass the selection to
  `_parameter_table` and `_metric_table`, keeping their column-spec computation
  on the selected cells only.

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
