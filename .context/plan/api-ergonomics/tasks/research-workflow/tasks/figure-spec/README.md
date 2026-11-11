---
type: task
title: Decline a separate FigureSpec grammar
description: "Declined because extending PlotSpec meets the known figures without a second grammar and schema."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/plotspec-extension/README.md"]
status: closed
resolution:
  reason: "Extending PlotSpec covers the known figures without a second grammar or persisted schema."
---

# Decline a separate FigureSpec grammar

## Goal

Proposal: replace `PlotSpec` with a layered `FigureSpec` grammar of data, layers, scales and evidence. Declined for now.

## Acceptance

- [x] Record the decision and its evidence.

## Why

`PlotSpec` already has two renderers, tests and a snapshot schema. `plotspec-extension` adds ribbons, overlays and trusted labels, which covers the known figures. A new grammar would duplicate that work and force another persisted format.

## Reopen if

At least two concrete required figures cannot be expressed after `plotspec-extension`.

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
- [Decide figure scales, panels, bars and orientation](../../../../../api-cohesion/tasks/report-review/tasks/figure-grammar/README.md)

<!-- insitu:end -->
