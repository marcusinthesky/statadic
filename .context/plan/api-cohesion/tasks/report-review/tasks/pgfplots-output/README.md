---
type: task
title: Make PGFplots output compact and restyleable
description: Write PGFplots coordinates at shortest round-trip precision, drop empty options, style series from the document's cycle list and add a direct save to .tex.
links:
  - .context/plan/api-cohesion/tasks/report-review/audit.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/latex-tables/README.md
status: done
---

# Make PGFplots output compact and restyleable

## Goal

`PlotSpec.to_pgfplots` writes seventeen-digit coordinates, empty axis options and
fixed colours. Row P06 of the [dispositions](../../../../proposal-dispositions.md)
accepts the repair. Start after `latex-tables` is done: both edit
`api/reporting.py`.

Retained coordinates never change. `PlotSpec` keeps its fields, and
`to_matplotlib` is not edited.

## Acceptance

- [x] Every coordinate, tick position and reference endpoint prints as the shortest text that reads back to the same float; an integral value prints without `.0`.
- [x] `xlabel`, `ylabel` and `title` are emitted only when nonempty.
- [x] No colour is fixed for a series: each uses `\addplot+`, so the document's cycle list styles it.
- [x] An interval's bars or ribbon take the style of their own series, and a ribbon fills with `fill=.`.
- [x] A figure mixing labelled and unlabelled series gives each a distinct style and a legend entry only to labelled ones.
- [x] `PlotSpec.save_pgfplots(path, *, width_cm=12, height_cm=7)` writes exactly `to_pgfplots(...)` as UTF-8.
- [x] The docs figure compile test covers a ribbon with two overlays, bars, and a labelled reference line beside an unlabelled series, from the cache with no Overfull box.
- [x] `tests/api/test_reporting.py`, `tests/api/test_plots.py` and the docs reporting tests pass; lint and strict typing pass for the touched files.

## Specification

- Coordinate text: `repr(float(value))` with a final `.0` removed.
- Series order is the base plot, then each overlay. For each series write its
  interval first, then the series itself.
- Interval: `\addplot+[no marks,forget plot]` per bar, or
  `\addplot+[draw=none,mark=none,fill=.,fill opacity=0.2,forget plot]` for the
  closed ribbon polygon. `forget plot` does not advance the cycle list, so the
  interval takes the entry of the series that follows.
- Labelled series: `\addplot+[no marks]` for a line, `\addplot+[only marks]`
  otherwise, then `\addlegendentry{...}`.
- Unlabelled series: the same with `forget plot`, so no legend entry is consumed.
  When another series follows, write `\pgfplotsset{cycle list shift=N}` after it,
  where `N` counts the unlabelled series so far. The audit compiled and inspected
  this pattern.
- Reference lines stay `\addplot[black,dashed,no marks]`, with `forget plot` when unlabelled.
- Tracked files under `src/latex/projects/*/src/generated/` are not regenerated here.

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
