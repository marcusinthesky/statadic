---
type: task
title: Declare assignment units and exchangeability blocks
description: "Let permutation plans name the randomized unit and exchangeability blocks, and enumerate distinct admissible label arrangements."
links: [".context/reference/hemerik-goeman-2018/README.md", ".context/reference/winkler-2015/README.md", ".context/reference/anderson-2001-permanova/README.md", ".context/reference/mantel-1967/README.md"]
status: done
---

# Declare assignment units and exchangeability blocks

## Goal

A permutation test is valid for the group of relabellings the design makes
equally likely. STATADIC exposed only free relabelling of rows. Four subjects
observed twice, two per arm, gave an exact p of 1/35 from 70 row
arrangements, where the randomization actually used admits six subject-level
arrangements and gives 1/3. Unequal cluster sizes could not be expressed at
all, because a cluster-level reassignment is not a row permutation.

## Decisions

- **API.** `PermutationPlan(unit=..., blocks=...)`. `unit` is the unit of
  assignment (all its rows carry one label and move together); `blocks` are
  exchangeability blocks (labels never cross a block). Each is a column of the
  fitted grouping `Observations`, resolved at inference like
  `Clustered("firm")`, or labels aligned with `row_ids`. Unset means free rows
  and keeps the native seeded streams.
- **Checks at inference.** Labels are constant within units; units nest in
  blocks; a declared design must admit more than one arrangement.
- **Group.** Uniform unit relabelling within each block. Exact plans enumerate
  each distinct arrangement once (∏ multinomials), with the same p-value as
  full-group enumeration because every arrangement has equal multiplicity.
  The budget is `max_exact_arrangements`.
- **Evidence.** DISCO and PERMANOVA retain `arrangements` (group code per
  observation per draw); PERMDISP keeps residual `schedules`. Inference
  retains `ExchangeabilityDesign` and the admissible count, reported as
  `exchangeability`. No new type-level provenance axis: the origin docstrings
  now name "the declared relabelling group".
- **Supplied schedules** with a declared design must lie in its orbit.
- **Formal scope.** STATADIC-RANDOMIZATION-001 and -002 are stated for any finite
  group acting with invariant weights, which covers these subgroups. The
  distinct-arrangement reduction (orbit–stabilizer) is reviewed, not
  formalized. No statement or pin changed.

## Deferred

- Whole-unit moves for PERMDISP residuals and Mantel objects. Both are row
  permutations, so a whole-unit move needs equal-size units and a within-unit
  correspondence; both reject `unit`.
- Whole-block shuffles (PALM's exchangeable blocks moved intact) and
  ordered within-unit designs such as series or grids.
- Names for Mantel design labels: Mantel keeps no `Observations`.

## Acceptance

- [x] `unit` and `blocks` for DISCO and PERMANOVA; `blocks` for PERMDISP and Mantel.
- [x] The 1/35 versus 1/3 release test, by name and by explicit labels.
- [x] Unequal unit sizes and within-block exact p-values match independent enumeration.
- [x] Design, orbit and budget errors are raised at inference.
- [x] Docs (`composition.md`, `resampling.md`) and the changelog describe the contract.

Tests: `src/python/packages/statadic/tests/api/comparison/test__shared.py`.

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
