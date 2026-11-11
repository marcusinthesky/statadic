---
type: task
title: Decide figure scales, panels, bars and orientation
description: Owner decision on extending PlotSpec with axis scales, orientation, bars and panels, now that four required figures cannot be expressed.
links:
  - .context/plan/api-cohesion/tasks/report-review/audit.md
  - .context/plan/api-ergonomics/tasks/research-workflow/tasks/figure-spec/README.md
  - .context/plan/api-ergonomics/tasks/research-workflow/tasks/plotspec-extension/README.md
status: backlog
---

# Decide figure scales, panels, bars and orientation

## Goal

The declined
[FigureSpec grammar](../../../../../api-ergonomics/tasks/research-workflow/tasks/figure-spec/README.md)
reopens when two required figures cannot be expressed. P10 in the
[audit](../../audit.md) names four:

- The software paper's coverage figure: four panels with a shared range, assembled by hand.
- Coverage or power against sample size on a logarithmic axis.
- A permutation null as a histogram with the observed statistic marked.
- A coefficient plot with names on the vertical axis, where long names fit.

Recommended: extend `PlotSpec` rather than add a second grammar. Add `xscale`
and `yscale`, an `orientation`, a `bar` kind with precomputed heights, and a
small grid record holding `PlotSpec` panels with shared axes. Each is a persisted
field, so they land together under one snapshot schema change. Panels need the
PGFplots `groupplots` library, which the vendored bundle lacks, so that part also
needs a bundle refresh.

Row P10 of the [dispositions](../../../../proposal-dispositions.md) leaves the
choice to the owner.

Owner: Backlog. Trigger: the owner accepts the extension or reopens the grammar.

## Acceptance

- [ ] The owner chooses between extending `PlotSpec` and a layered grammar.
- [ ] Each accepted element states its fields, its PGFplots and Matplotlib rendering, and its snapshot form.
- [ ] Bar heights and bin edges are computed by a named function outside the renderer.
- [ ] The four figures above compile from the pinned bundle after one refresh.

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
