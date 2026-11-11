---
type: plan
title: Prove one coherent research workflow before release
description: "Validate the researcher's journey, deliver one verified publication bundle, then freeze the scientific and compatibility contracts. Publication stays on hold."
links: [".context/plan/research-release/handoff.md", ".context/plan/research-release/evidence.md", ".context/plan/publishing-companion/README.md", ".context/plan/release-experience/publishing-design.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md"]
status: in-progress
---

# Prove one coherent research workflow before release

## Goal

Apply the decisions from the 2026-11-06 assessment of the release-priorities proposal. Make the existing pieces work as one trustworthy research system, rather than adding plotting, dataframe or checkpoint ecosystems. Library publication stays on hold by owner decision.

## Acceptance

- [x] Complete [walkthroughs](tasks/walkthroughs/README.md) and [artifact-contract](tasks/artifact-contract/README.md).
- [x] Complete the `bundle`, `paper`, `operations` and `replication` tasks of the [publishing companion](../publishing-companion/README.md).
- [x] Add the release-gate items below to [publication-hold](../api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md) as each one becomes checkable.
- [x] Keep [plot-accessor](tasks/plot-accessor/README.md) blocked until its trigger is met.
- [x] Pass `just check` after the final task and report any check that could not run.

## Start here

Read the [handoff guide](handoff.md) first. The [evidence](evidence.md) records what the assessment found at `0d68571`.

| Order | Work | Depends on |
| --- | --- | --- |
| 1 | [walkthroughs](tasks/walkthroughs/README.md) | None |
| 2 | [artifact-contract](tasks/artifact-contract/README.md) | None; may run beside 1 |
| 3 | Companion [bundle](../publishing-companion/tasks/bundle/README.md) | None |
| 4 | Companion [paper](../publishing-companion/tasks/paper/README.md) | 1 for `residual_plot`, 3 |
| 5 | Companion [operations](../publishing-companion/tasks/operations/README.md) | 3, 4 |
| 6 | Companion [replication](../publishing-companion/tasks/replication/README.md) | 5 |

The companion's reviewer-revision task stays in backlog until this slice is done.

## Release gate

The release gate is scientific, not cosmetic. It lives in [publication-hold](../api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md), which gains two items from this plan:

- The walkthrough lane passes on the frozen wheel.
- The owner confirms the artifact compatibility promise.

Do not present a publication-native workflow as shipped until the companion slice passes replication.

## Deferred or unchanged

- [plot-accessor](tasks/plot-accessor/README.md) is blocked on walkthrough evidence.
- `statistical-table`, `prediction-bands` and `labelled-arrays` stay blocked; `holoviews-stack` and `orbax-checkpoints` stay closed. The assessment agreed with each record, so none changed.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| in-progress | [Prepare the STATADIC alpha release](tasks/alpha-readiness/README.md) | Repair the reviewed stack, qualify downstream-required contracts and retain a scoped alpha release bundle. |
| blocked | [Defer a fitted-result plot accessor](tasks/plot-accessor/README.md) | Deferred until the walkthroughs show that the plot helpers read poorly in practice. |

`2 open; 3 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Optional reproducible research publishing companion](../publishing-companion/README.md)
- [Correct advertised guarantees, then prove the first ones](../release-guarantees/README.md)

<!-- insitu:end -->
