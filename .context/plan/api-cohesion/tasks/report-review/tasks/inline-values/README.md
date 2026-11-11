---
type: task
title: Export report numbers as manuscript value macros
description: Let a report write its numbers as named TeX values in a self-contained file, so prose cites an estimate without a hand-written binding.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/inline-values/specification.md
  - .context/plan/api-cohesion/tasks/report-review/audit.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/test-tables/specification.md
status: done
---

# Export report numbers as manuscript value macros

## Goal

A manuscript quotes estimates in prose as well as in tables. The software paper
binds 92 values through 53 hand-written calls: see P12 in the
[audit](../../audit.md). Each repeats a format string and a name, and nothing ties
the name to the report row it came from.

The owner accepted value export on 2026-11-11 (row P12 of the
[dispositions](../../../../proposal-dispositions.md)). The file is self-contained:
it defines `\jcvalue` and `\jcnum` itself when they are missing, so it works in
any manuscript and beside the repository's own style. The
[specification](specification.md) is normative. Start when `test-tables` is done,
because both edit `api/reporting.py` and values include test rows.

## Acceptance

- [x] `Report.to_values(*, prefix, digits=6, decimals=None)` returns the file text and `Report.save_values(path, ...)` writes it.
- [x] Every available parameter, metric and test quantity is declared once under a deterministic key.
- [x] Two quantities that would share a key raise `ValueError` naming both.
- [x] Unavailable quantities are not declared; each is listed in a comment with its reason.
- [x] Numbers follow the table cell rules; p-values carry their relation, `= 0.497` or `< 0.001`.
- [x] A docs compile test inputs an exported file, with and without `statadic-manuscript.sty`, and uses `\jcnum`, `\jcvalue` and a p-value.
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

_No declared backlinks._

<!-- insitu:end -->
