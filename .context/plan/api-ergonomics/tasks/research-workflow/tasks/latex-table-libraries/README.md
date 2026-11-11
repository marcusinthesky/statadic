---
type: task
title: Decline third-party LaTeX table libraries
description: "Declined because the table problems are statistical content, not formatting a library would solve."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/latex-report/README.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/statistical-table/README.md"]
status: closed
resolution:
  reason: "The problems are statistical content and layout decisions, which a generic table formatter would not solve."
---

# Decline third-party LaTeX table libraries

## Goal

Proposal: depend on `tabulate`, `latextable`, `PyLaTeXtables` or `PyTabular`. Declined for now.

## Acceptance

- [x] Record the decision and its evidence.

## Why

The defects are empty columns, misplaced metadata, no stated interval level and page overflow. `latex-report` fixes those inside a small deterministic renderer that already escapes labels and preserves unavailable values. A generic formatter would not supply statistical semantics. Ideas such as multi-model layouts are retained under `statistical-table`.

## Reopen if

A concrete layout proves too costly to maintain in the STATADIC renderer.

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

_No declared backlinks._

<!-- insitu:end -->
