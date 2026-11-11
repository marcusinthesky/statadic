---
type: task
title: Apply the assessed research-workflow decisions
description: "Implement, defer or decline each assessed research-workflow proposal, with publication on hold."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/execution.md", ".context/plan/release-experience/publishing-design.md"]
status: in-progress
---

# Apply the assessed research-workflow decisions

## Goal

Apply the decisions from the 2026-11-05 assessment of proposed research-workflow features, so each proposal is implemented, deferred or declined explicitly. Library publication stays on hold by owner decision.

## Acceptance

- [x] Complete the eleven implementation sub-tasks using the dependencies below.
- [x] Complete the package-wide [dead-code cleanup](tasks/dead-code-cleanup/README.md).
- [ ] Keep deferred sub-tasks blocked until their stated conditions are met.
- [ ] Leave declined sub-tasks closed unless the owner reopens one.
- [x] Pass `just check` after the final implementation sub-task and report any check that could not run.

The implementation and cleanup are complete; see [verification](verification.md). This task stays `in-progress` because its blocked children remain open. Follow-on release work is in the [research-release plan](../../../research-release/README.md).

## Start here

Read the [handoff guide](handoff.md) first. Measurements are in [evidence](evidence.md). The [execution guide](execution.md) assigns parallel work and review gates; the numbered dependencies below remain authoritative for feature prerequisites.

| Order | Sub-task | Depends on |
| --- | --- | --- |
| 1 | [reject-narrowing](tasks/reject-narrowing/README.md) | None |
| 2 | [prediction-roles](tasks/prediction-roles/README.md) | None |
| 3 | [mapping-inputs](tasks/mapping-inputs/README.md) | 1, 2 |
| 4 | [table-inputs](tasks/table-inputs/README.md) | 3 |
| 5 | [predictor-naming](tasks/predictor-naming/README.md) | After 2, to limit conflicts |
| 6 | [require-valid](tasks/require-valid/README.md) | None |
| 7 | [latex-report](tasks/latex-report/README.md) | None |
| 8 | [pgfplots-gate](tasks/pgfplots-gate/README.md) | None |
| 9 | [plotspec-extension](tasks/plotspec-extension/README.md) | 8 |
| 10 | [plot-helpers](tasks/plot-helpers/README.md) | 3, 7, 9 |
| 11 | [linear-mmd-refinement](tasks/linear-mmd-refinement/README.md) | None |

## Deferred

These are `blocked`. Do not start them until their acceptance conditions can be met:

- [statistical-table](tasks/statistical-table/README.md)
- [prediction-bands](tasks/prediction-bands/README.md)
- [labelled-arrays](tasks/labelled-arrays/README.md)
- [publication-hold](tasks/publication-hold/README.md)

## Declined for now

These are `closed`, and each record names its reopening condition:

- [from-formula](tasks/from-formula/README.md)
- [orbax-checkpoints](tasks/orbax-checkpoints/README.md)
- [latex-table-libraries](tasks/latex-table-libraries/README.md)
- [core-publication-object](tasks/core-publication-object/README.md)
- [holoviews-stack](tasks/holoviews-stack/README.md)
- [figure-spec](tasks/figure-spec/README.md)

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| blocked | [Defer pandas ExtensionArray and xarray integration](tasks/labelled-arrays/README.md) | Deferred until a labelled-output or measure-per-cell use case exists; Arrow input already covers ordinary pandas frames. |
| blocked | [Defer confidence bands on fitted curves](tasks/prediction-bands/README.md) | Deferred because bands on fitted curves are a new statistical capability needing their own contract and qualification. |
| blocked | [Hold the software paper and public release](tasks/publication-hold/README.md) | Publication is on hold by owner decision. Records what must be true before any paper or public release. |
| blocked | [Defer StatisticalTable and rate-ratio tables](tasks/statistical-table/README.md) | Deferred until the report fix lands and a real multi-model or rate-ratio table is needed. |

`4 open; 18 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Apply the composition-review waves](../composition-waves/README.md)

<!-- insitu:end -->
