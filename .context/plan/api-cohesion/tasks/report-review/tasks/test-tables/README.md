---
type: task
title: Give test outcomes their own typed report rows
description: Add a typed TestRow section to Report, render it as a table of tests in text, HTML and LaTeX, and persist it in analysis snapshots.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/test-tables/specification.md
  - .context/plan/api-cohesion/tasks/report-review/audit.md
status: done
---

# Give test outcomes their own typed report rows

## Goal

A test outcome reaches its report as metrics named `name.statistic`,
`name.pvalue` and `name.component`: see P09 in the [audit](../../audit.md). A
manuscript sets tests as one row each, with statistic, reference law and p-value
in columns. Formatting by metric name would be guessing, so a rank prints as
`1.000` and an underflowed p-value as `0.000`.

The owner accepted typed test rows on 2026-11-11 (row P09 of the
[dispositions](../../../../proposal-dispositions.md)). This task adds the record,
its renderers and its snapshot form. [Test producers](../test-producers/README.md)
then fills it from every calibrated test. The [specification](specification.md)
is normative.

## Acceptance

- [x] `statadic.api.reporting.TestRow` exists with the fields and validation of the specification, and `Report` retains `tests`.
- [x] `to_text`, `to_html` and `to_latex` render one row per test: name, statistic, reference law and p-value.
- [x] LaTeX sets the law as math (`$\chi^2(2)$`, `$F(1, 78)$`) and the p-value with the `<0.001` floor; Reference and $p$ columns are omitted when no row has a p-value.
- [x] An inference report's Statistic header reads `$z$` for the normal reference and `$t$` for the t reference.
- [x] Snapshots round-trip test rows, and the artifact schema version moves from 6 to 7 once.
- [x] The docs report compile test covers a test table with chi-square, F, resampled and unavailable rows.
- [x] `tests/api/test_reporting.py`, `tests/api/test_artifacts.py`, `tests/api/test_ordination_artifacts.py` and the docs reporting tests pass; lint and strict typing pass for the touched files.

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

- [Enforce report availability states and exact reference-law parameters](../../../final-review/tasks/report-record-contract/README.md)

<!-- insitu:end -->
