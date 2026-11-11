---
type: task
title: Move the software paper to exported report values
description: Replace the paper generator's hand-written report bindings with exported values and regenerate every tracked output from a committed tree.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/inline-values/specification.md
status: done
---

# Move the software paper to exported report values

## Goal

`src/latex/projects/01_statadic/analysis/generate.py` binds the OLS and Poisson report
numbers by hand, and the tracked outputs under `src/generated/` still use the old
table and figure format.

Regeneration records the base revision and whether the package source is modified,
and it reruns the coverage study. Run it only from a committed tree, so the record
matches the code. Changing the generator or chapter keys without regenerating would
break the tracked build, so both happen together.

The trigger, a commit of the Report review changes, was met by `6a3d36c`. Regeneration also found that the generator and both printed listings used constructors removed in earlier API waves (`OLS`, `Conditional`, `LogRate`); they now use `Regression`. Every report number, agreement difference and coverage value is unchanged; only the machine-dependent timing ratios moved.

## Acceptance

- [x] The generator binds report numbers with `Report.save_values` (or `to_values`) and keeps hand-written bindings only for numbers that are not report values.
- [x] Chapters use the exported keys, `\jcnum` for numbers and `$p \jcvalue{...}$` for p-values.
- [x] `just latex::generate 01_statadic` runs from a committed tree, and the tracked tables, figures and values use the new formats.
- [x] `just latex::check` passes and the arXiv bundle recompiles cleanly.

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
