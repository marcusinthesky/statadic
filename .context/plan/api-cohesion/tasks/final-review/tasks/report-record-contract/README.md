---
type: task
title: Enforce report availability states and exact reference-law parameters
description: Render Normal reference parameters without rounding them away, and reject report records whose availability declarations contradict their values.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/test-tables/README.md
status: done
---

# Enforce report availability states and exact reference-law parameters

## Goal

This task covers review findings R3 and R4, accepted as rows F03 and F04, and the
stale schema docstrings (row F05).

- **R3.** `_count_text` formats non-integers with `:.2f`. `_reference_cell` uses it
  for Normal location and scale, so `Normal(-0.0001, 0.001)` renders as
  `N(-0.00, 0.00²)`. That misstates the reference law.
- **R4, intervals.** `ParameterRow` rejects endpoints only when
  `interval_unavailable_reason` is set, so `interval_status="unavailable"` with
  endpoints passes. `Report.require_available("interval")` checks only the
  endpoints.
- **R4, metrics.** A `Metric` with `available=False` and a finite value is
  accepted. Tables show it, while the value export withholds it.

Built-in producers already comply: unavailable metrics pass `None`, and
`api/spatial/resampling.py` gives reasons. Tightening therefore rejects only
hand-built contradictions. A partial result keeps its point estimate when its
uncertainty is unavailable.

## Acceptance

- [x] Normal reference parameters keep significant digits, so no positive scale renders as zero, in LaTeX, HTML and text.
- [x] Contradictory interval and metric states are rejected at construction.
- [x] `require_available("interval")` and the value export follow the declared interval status.
- [x] A state matrix covers validation, `require_available`, text, HTML, LaTeX, value export and a schema-7 snapshot round trip, with legitimate partial-result controls.
- [x] The docs compile test asserts both that a small-parameter Normal row compiles and its source.
- [x] Artifact module docstrings name schema version 7.
- [x] The named test files pass; lint passes for the touched files.

## Specification

- Files: `src/statadic/api/reporting.py`; line 1 of `src/statadic/api/artifacts/_schema.py`
  and `_store.py`; `tests/api/test_reporting.py`; `tests/api/test_artifacts.py`;
  `src/python/apps/docs/tests/test_reporting_workflow.py`.
- R3:
  - `_reference_cell(row, digits)` receives the table's `digits` from
    `_test_table`.
  - Normal parameters render as `\num{<text>}`, where text is
    `format(value, f".{digits}g")` with the exponent normalised (`1e-05` becomes
    `1e-5`, `1e+06` becomes `1e6`). An integral value stays exact.
  - `(0, 1)` stays `$N(0, 1)$`. A general row is
    `$N(\num{-0.0001}, \num{0.001}^2)$`.
  - Chi-square and F degrees of freedom keep `_count_text`.
  - Check `to_html` and `to_text` for the same rounding, and fix them the same way
    if they round.
- R4, intervals:
  - Add a property `ParameterRow.interval_available`, true when both endpoints are
    present and `interval_status in (None, "available")`.
  - In `__post_init__`, an interval is unavailable when `interval_unavailable_reason`
    is set, or when `interval_status` is not `None` and not `"available"`.
  - An unavailable interval must have no endpoints and a reason.
  - `interval_status == "available"` requires both endpoints.
  - `require_available("interval")` and `_value_entries` use `interval_available`,
    with missing-interval entries keyed on the same rule.
- R4, metrics: `Metric.__post_init__` rejects `available=False` with a non-`None`
  value, mirroring `TestRow`.
- Keep valid: an estimate present with `valid=False`, a reason and no standard
  error. Its estimate renders and exports; `require_available("std_error")` fails.
- Search `src/` and `src/python/apps/` for constructors that would now fail. Report
  any you find instead of weakening the rule.
- Docstrings: replace "version-six" with "version-7".
- Docs compile test:
  - Add `TestRow("location shift", 1.5, 0.13, law="normal", law_parameters=(-0.0001, 0.001))`,
    or a row of the same shape, to a compiled fixture.
  - Assert the exact `$N(\num{-0.0001}, \num{0.001}^2)$` source alongside
    successful compilation.

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
