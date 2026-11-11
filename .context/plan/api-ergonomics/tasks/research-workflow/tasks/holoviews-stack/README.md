---
type: task
title: Decline HoloViews with Matplotlib PGF as the primary stack
description: "Declined as primary stack because its PGF route fails in the pinned environment and is not PGFplots; an optional adapter stays possible."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: closed
resolution:
  reason: "Its PGF route needs xelatex, absent from the pinned toolchain, and produces low-level drawing commands rather than editable PGFplots."
---

# Decline HoloViews with Matplotlib PGF as the primary stack

## Goal

Proposal: build plotting on HoloViews and produce PGF through Matplotlib. Declined as the primary stack for now.

## Acceptance

- [x] Record the decision and its evidence.

## Why

- Matplotlib PGF raises `'xelatex' not found` here, directly and through HoloViews, and leaves a truncated file. The pinned TeX toolchain is Tectonic.
- Its output is low-level drawing commands, not editable PGFplots.
- HoloViews adds 38 packages and 318 MB, including pandas, Bokeh and Panel.
- Native `PlotSpec.to_pgfplots()` compiles with the pinned bundle and stays editable.

Matplotlib stays available through `to_matplotlib`, and PDF output can reach a manuscript through `\includegraphics`. Adding `xelatex` was not tested and would add a second TeX toolchain.

## Reopen if

The owner wants interactive dashboards. Then build an optional adapter over `PlotSpec`, outside the numerical core, without changing the publication path. HoloViews accepts plain tuples and resolves against the lock.

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
