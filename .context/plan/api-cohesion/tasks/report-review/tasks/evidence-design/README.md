---
type: task
title: Design how method evidence reaches a manuscript
description: Recommend how a result's citations, reference comparisons, simulation studies and Lean statements reach a LaTeX manuscript, for the owner to decide.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/evidence-design/design.md
  - .context/plan/api-cohesion/tasks/report-review/audit.md
  - .context/plan/release-experience/publishing-design.md
  - .context/plan/api-ergonomics/tasks/research-workflow/tasks/core-publication-object/README.md
status: done
---

# Design how method evidence reaches a manuscript

## Goal

The owner's aim pairs each method with its reference comparison, simulation study
and Lean statement, and lets a researcher cite the original research, all inside
a LaTeX manuscript. Today none of that is reachable from a fitted result or a
`Report`: see P13 in the [audit](../../audit.md). Row P13 of the
[dispositions](../../../../proposal-dispositions.md) asks for a design before code.

This task writes `design.md` beside this record. It changes no code. The
[accepted publishing design](../../../../../release-experience/publishing-design.md)
and the declined
[core Publication object](../../../../../api-ergonomics/tasks/research-workflow/tasks/core-publication-object/README.md)
constrain it: `statadic-publishing` consumes snapshots and `statadic` never depends on it.

The [design](design.md) recommends a generated evidence index, keyed by catalogue
variant and shipped with `statadic-publishing`. Its section 6 lists six decisions that
remain with the owner; nothing is built until they are made.

## Acceptance

- [x] Inventory what exists per method variant, with counts: citation keys, reference mappings, study mappings, Lean claims, and which of these ship in the wheel.
- [x] Trace three concrete analyses end to end (OLS with HC3, Poisson with exposure, one permutation test) and state what a researcher must type by hand today to cite and qualify each.
- [x] Describe at most three options for carrying evidence from a result to a manuscript, each with the data shape, where it lives, what ships, and what it costs to keep true.
- [x] Recommend one option and name the first step small enough to verify alone.
- [x] State what the design must never claim: a citation is not source fidelity, a reference match is not calibration, and a Lean statement covers only its stated assumptions.
- [x] List the owner decisions the recommendation depends on.

## Specification

- Sources: `src/python/apps/studies/catalogue.json`, `src/lean/registry/`,
  `src/latex/refs.bib`, `.context/reference/`, `src/python/packages/statadic-evidence/`,
  `src/python/packages/statadic-publishing/`, `src/latex/templates/statadic-manuscript.sty`
  and the Lean macro file `src/latex/projects/01_statadic/src/generated/statadic-mathematics.tex`.
- Keep `design.md` under 200 lines. Prefer a table per option to prose.
- Write no private path, user name or downstream repository name into it.

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
