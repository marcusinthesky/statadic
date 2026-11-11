---
type: task
title: Defer StatisticalTable and rate-ratio tables
description: "Deferred until the report fix lands and a real multi-model or rate-ratio table is needed."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/latex-report/README.md"]
status: blocked
---

# Defer StatisticalTable and rate-ratio tables

## Goal

Record the deferred idea of a presentation layer between `Report` and renderers, for multi-model tables and rate ratios. Do not implement it now.

## Acceptance

- [ ] `latex-report` is done.
- [ ] The owner or a `statadic-publishing` consumer names a concrete table that `to_latex` cannot express.
- [ ] The owner decides whether the code belongs in core STATADIC or `statadic-publishing`.

## Constraints when unblocked

- It must be an immutable view over `Report`, with no statistical computation in renderers.
- Rate ratios must come from a typed model family, never from a title or name.
- Exponentiate the estimate and interval endpoints together. Omit the standard error unless a transformed one has been qualified.
- Canonical log-scale values stay retained.

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

- [Review and repair the Report API](../../../../../api-cohesion/tasks/report-review/README.md)
- [Decline third-party LaTeX table libraries](../latex-table-libraries/README.md)

<!-- insitu:end -->
