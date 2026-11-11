---
type: task
title: Render report tables with aligned numbers and exact counts
description: Make Report.to_latex a decimal-aligned booktabs and siunitx table with exact counts, display labels and a direct save to .tex.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/latex-tables/specification.md
  - .context/plan/api-cohesion/tasks/report-review/audit.md
status: done
---

# Render report tables with aligned numbers and exact counts

## Goal

`Report.to_latex` misstates counts, strips trailing zeros, right-aligns ragged
numbers and labels rows with identifiers. Rows P01–P03 of the
[dispositions](../../../../proposal-dispositions.md) accept the repair. The
[specification](specification.md) is normative; the [audit](../../audit.md) holds
the measurements.

Retained report values never change. Only formatting does.

## Acceptance

- [x] An `int` value prints exactly in LaTeX, text and HTML at any `digits`.
- [x] Numeric columns are siunitx `S` columns whose `table-format` is computed from the cells; trailing zeros are kept.
- [x] `decimals=` fixes decimal places; `digits=` keeps significant digits and stays the default.
- [x] An unavailable cell is a dash and its reason stays a note.
- [x] Retained `confidence` heads both interval columns as one spanning header.
- [x] Parameter p-values print to three decimals with a `<0.001` floor.
- [x] Metrics close the parameter table as footer rows; a metrics-only report keeps its own table.
- [x] `labels=` supplies display labels, plain by default and TeX when `label_markup="tex"`.
- [x] Non-ASCII text the pinned fonts lack is written as TeX, using no font-size command.
- [x] `notes=False` omits the trailing notes.
- [x] `Report.save_latex(path, ...)` writes exactly `to_latex(...)` as UTF-8.
- [x] The docs report template loads `siunitx`, and its compile test covers every case above and rejects missing characters.
- [x] `tests/api/test_reporting.py` and the docs reporting tests pass; lint and strict typing pass for the touched files.

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

- [Make PGFplots output compact and restyleable](../pgfplots-output/README.md)

<!-- insitu:end -->
