---
type: task
title: Extend PlotSpec with ribbons, overlays and trusted labels
description: "Add ribbons, overlays, legend labels and explicitly trusted TeX labels to PlotSpec without adding a new figure grammar."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/pgfplots-gate/README.md"]
status: done
---

# Extend PlotSpec with ribbons, overlays and trusted labels

## Goal

Extend `PlotSpec` just enough for publication figures: uncertainty ribbons, several series on one axis, legend labels and math in labels. It remains immutable, precomputed and renderer-independent.

## Acceptance

- [x] New fields: `label: str = ""`, `interval_style: Literal["bars", "ribbon"] = "bars"`, `overlays: tuple[PlotSpec, ...] = ()` and `markup: Literal["plain", "tex"] = "plain"`.
- [x] A ribbon requires `kind="line"`, strictly increasing `x` and the existing `interval_meaning`.
- [x] Overlays share the parent's axes: their axis labels and title are empty, they cannot nest, and they use the parent's `markup`.
- [x] PGFplots draws ribbons as a closed polygon with `coordinates {...} -- cycle;` and needs no extra library.
- [x] With `markup="plain"`, labels are escaped in PGFplots and Matplotlib uses `parse_math=False`. With `"tex"`, labels pass through verbatim, newlines are rejected, and `$\hat\beta_1$` renders in both.
- [x] Matplotlib draws the same data, using `fill_between` for ribbons and a legend when any label exists.
- [x] Snapshots round-trip all fields. `load_analysis(path, *, trust_tex=False)` raises `ArtifactError` on `tex` markup unless trusted.
- [x] The compile gate covers a ribbon, an overlay and a trusted label.
- [x] Docs and `CHANGELOG.md` are updated, and `just check` passes.

Depends on `pgfplots-gate`.

## Specification

- Main code: `PlotSpec`, `_tex` and `render_pgfplots` in `api/reporting.py`; `_report_state`, `_restore_report` and `load_analysis` in `api/artifacts/_snapshot.py`.
- Snapshot records use exact keys from `fields(PlotSpec)`, so handle overlays recursively and apply the schema rule from the handoff guide.
- `trust_tex` exists because re-rendering a snapshot from disk could otherwise inject arbitrary TeX.
- The plain-mode Matplotlib change fixes a real disagreement: today Matplotlib parses `$...$` while PGFplots prints it literally.

## Non-goals

No facets, themes, secondary axes, log scales, external data files, `FigureSpec`, HoloViews or computing any plotted value.

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

- [Decide figure scales, panels, bars and orientation](../../../../../api-cohesion/tasks/report-review/tasks/figure-grammar/README.md)
- [Decline a separate FigureSpec grammar](../figure-spec/README.md)
- [Add coefficient-interval and prediction-line helpers](../plot-helpers/README.md)
- [Defer confidence bands on fitted curves](../prediction-bands/README.md)

<!-- insitu:end -->
