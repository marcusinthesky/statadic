---
type: plan
title: Correct advertised guarantees, then prove the first ones
description: "Remove the familywise, non-finite p-value and equivalence guarantees STATADIC lacks and keep DOIs in the bibliography, then prove sampled permutations, weighted V-MMD, WLS and HC algebra, and Euclidean negative type."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md", ".context/plan/formal-literature/tasks/literature-proofs/README.md", ".context/plan/research-release/README.md"]
status: in-progress
---

# Correct advertised guarantees, then prove the first ones

## Goal

Before release, no STATADIC docstring or page should promise a statistical guarantee the code does not have, and the first headline guarantees should have reviewed Lean proofs.

A review on 2026-11-06 found three advertised guarantees that fail:

- the adaptive familywise correction rejects a true null 7.25% of the time at a nominal 5%;
- a NaN permutation statistic produces p = 0.001;
- the equivalence test claims a Type I guarantee from a theorem that does not cover it.

The [evidence](evidence.md) has the measurements.

The plan changes behaviour only where a guarantee is wrong. It adds no new scientific procedure, Lean package, dependency or result field.

## Acceptance

- [ ] The [corrections](tasks/corrections/README.md) and [proofs](tasks/proofs/README.md) tasks are complete.
- [x] No public docstring or documentation page claims familywise, Type I, size or coverage control that neither the code nor a registered claim supports.
  Keyword sweep on 2026-11-07 over `src/statadic` docstrings, the package docs and
  README: the `sharpe` package docstring and `studentized.py` overstated control
  or asymptotic validity and were softened; the rest are disclaimers or cite
  Holm (1979). A sweep is not a line-by-line review.
- [ ] Every new or changed Lean claim passed [claim review](tasks/proofs/tasks/claim-review/README.md) before the coordinator re-pinned statements.
- [ ] `just lean::certify` passes, with only `propext`, `Classical.choice` and `Quot.sound`.
- [ ] `just check` passes. Report any check that could not run.

## Start here

Read the [handoff](handoff.md) first. The [evidence](evidence.md) records what was measured and how to reproduce it. The coordinator records baseline and checkpoint results in [verification](verification.md).

| Wave | Work | Depends on |
| --- | --- | --- |
| 1 | [multiplicity](tasks/corrections/tasks/multiplicity/README.md) | Baseline |
| 1 | [nonfinite-pvalues](tasks/corrections/tasks/nonfinite-pvalues/README.md) | Baseline |
| 1 | [equivalence](tasks/corrections/tasks/equivalence/README.md) | Baseline |
| 1 | [sampled-permutation](tasks/proofs/tasks/sampled-permutation/README.md) | Baseline |
| 1 | [weighted-mmd](tasks/proofs/tasks/weighted-mmd/README.md) | Baseline |
| 1 | [linear-covariance](tasks/proofs/tasks/linear-covariance/README.md) | Baseline |
| 1 | [negative-type](tasks/proofs/tasks/negative-type/README.md) | Baseline |
| 2 | [sources](tasks/corrections/tasks/sources/README.md) | Wave 1 frozen |
| 2 | [claim-review](tasks/proofs/tasks/claim-review/README.md) | Proof lanes frozen and registry drafts merged |

Wave 1 lanes own disjoint files. Lean builds and full test runs are serialized as the handoff describes.

## Out of scope

- The externally proposed 28-package Lean layout.
- Transport, GMM/IV, GLM, GEL, panel, spatial, survival, bootstrap, Sharpe and cone proofs.
- Implementing Liu and Gandy's multiplier bootstrap, or any new adaptive multiplicity procedure.
- A Lake dependency on StatLean or MathEconLib. Workers may read them. Copying code needs the coordinator to confirm the licence and record provenance.
- Changing the `statadic-publishing` example's `formal = "unavailable"`. A proof of the HC formula does not establish interval coverage.
- Human semantic review of registered claims, which remains an owner action;
  the live Lean registry owns the roster rather than this historical plan.
- Any push, tag, upload or submission. Publication stays on hold.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| in-progress | [Prove the first release guarantees in Lean](tasks/proofs/README.md) | Four finite exact-real proof lanes (sampled permutations, weighted V-MMD, WLS and HC algebra, Euclidean negative type), then an independent review before any statement is re-pinned. |

`1 open; 1 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
