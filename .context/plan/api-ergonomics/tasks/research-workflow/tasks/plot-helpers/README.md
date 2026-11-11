---
type: task
title: Add coefficient-interval and prediction-line helpers
description: "Add two plot builders that need no new statistics: coefficient intervals from a report and prediction lines from a predictor."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/mapping-inputs/README.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/latex-report/README.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/plotspec-extension/README.md"]
status: done
---

# Add coefficient-interval and prediction-line helpers

## Goal

Replace hand-built coordinate tuples for the two common plots that need no new statistics: coefficient intervals from a retained report and a prediction line from a fitted or reloaded predictor.

## Acceptance

- [x] `Report.coefficient_plot(*, reference_value=0.0)` builds a `coefficient_interval` `PlotSpec` from retained rows, with parameter names as labels.
- [x] It raises `ValueError` when a row has no interval or metadata has no `confidence`. Its `interval_meaning` states level, reference and covariance.
- [x] `prediction_line(predictor, x, values, *, at=None, kind=None, label="", xlabel=None, ylabel=None, title="")` in new module `api/plots.py` returns a line `PlotSpec`.
- [x] It gives identical coordinates for live and reloaded predictors, and they match a direct `predict` call.
- [x] Nonfinite predictions raise a clear error instead of plotting.
- [x] `tests/api/test_plots.py`, Tach, inventory, catalogue, docs and `CHANGELOG.md` are updated, and `just check` passes.

Depends on `mapping-inputs`, `latex-report` (for `confidence`) and `plotspec-extension` (for `label`).

## Specification

- `coefficient_plot` lives in `api/reporting.py`, which depends only on `statadic.core`.
- `prediction_line` calls `predictor.predict({x: values, **(at or {})}, kind=kind)`, omitting `kind` when `None`. It accepts any object with that method, typed by a small protocol.
- Convert arrays with `float(value)` for tuple coordinates. `.to_numpy` is forbidden.
- Example: `prediction_line(fit, "load", grid, kind="rate")`.

## Non-goals

No `fit.plot.*` accessors, confidence bands, residual diagnostics or new statistics. Bands belong to `prediction-bands`.

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
