---
type: task
title: Review and repair the Report API
description: Audit Report tables, figures and exports against manuscript practice, repair confirmed defects and record the decisions left to the owner.
links:
  - .context/plan/api-cohesion/tasks/report-review/audit.md
  - .context/plan/api-ergonomics/tasks/research-workflow/tasks/statistical-table/README.md
  - .context/plan/api-ergonomics/tasks/research-workflow/tasks/figure-spec/README.md
status: in-progress
---

# Review and repair the Report API

## Goal

Report is the fifth workflow API and the last to be reviewed. The owner's aim,
stated on 2026-11-10: analyses enter a LaTeX manuscript as publication-quality
tables and PGFplots figures, saving a table to `.tex` is a crucial step, and the
reference comparisons, simulation studies and Lean statements behind a method are
usable in that manuscript.

The [audit](audit.md) measures current output. Rows P01–P13 of the
[dispositions](../../proposal-dispositions.md) own the decisions. Each phase ran
in waves of delegated tasks on disjoint files; the coordinator owned inventories,
the catalogue, `just sync` and one `just check` per phase.

## Acceptance

- [x] Phase 1, the repairs: `latex-tables`, `report-content`, `pgfplots-output`, `manuscript-integration` and `evidence-design`, with no persisted field change and no new statistic.
- [x] Phase 2, the owner's decisions of 2026-11-11 (siunitx required; coefficient tests, typed test rows and value export accepted): `test-tables`, `coefficient-tests`, `test-producers`, `inline-values`, `reporting-docs` and `paper-values`.
- [x] Phase 3, the owner's fit-statistics decision of 2026-11-11 (R² stating whether it is adjusted; log likelihood, AIC, BIC and HQIC with their original citations): `table-content`, `fit-labels`, `likelihood-statistics`, `spatial-statistics` and `fit-docs`.
- [x] Reconcile the public inventory and study catalogue after each phase.
- [x] Pass `just check` once after each phase and report any check that could not run.

All three phases passed `just check` on 2026-11-11. This task stays `in-progress`
while `figure-grammar` remains an owner decision.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| backlog | [Decide figure scales, panels, bars and orientation](tasks/figure-grammar/README.md) | Owner decision on extending PlotSpec with axis scales, orientation, bars and panels, now that four required figures cannot be expressed. |

`1 open; 16 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
