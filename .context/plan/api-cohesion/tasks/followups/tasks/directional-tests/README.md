---
type: task
title: Design directional coefficient tests
description: Specify a one-sided test for a single linear restriction, or document the alternative slot as reserved.
links:
  - .context/plan/api-cohesion/tasks/followups/tasks/directional-tests/design.md
status: backlog
---

# Design directional coefficient tests

## Goal

`Hypothesis.alternative` is public, and every `.test` rejects a stated value:
coefficient, dyadic, PERMDISP and Mantel. Read at `00e7d8f`:

- No Proposition can state a direction.
- `InferenceResult.test` offers only the Wald chi-square and F references, which
  are two-sided for one restriction.
- `Normal` is produced only by the Moran diagnostic.

A one-sided coefficient test therefore needs a design, not a switch. Row A26 in
the [dispositions](../../../../proposal-dispositions.md) separates it from
boundary and chi-bar laws, which stay deferred under A20.

The proposal and its reserved-slot alternative are in the [design](design.md).

Owner: Backlog. Trigger: a workflow that needs a directional coefficient test.
See [deferred work](../../deferred.md).

## Acceptance

- [ ] Decide: support a directional alternative for one linear restriction, or document the slot as reserved.
- [ ] If supported: a directional proposition, a signed studentized statistic and its normal or t reference; joint restrictions stay rejected.
- [ ] Verify against an independent one-sided calculation, including unavailable covariance.

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

- [Directional coefficient test design](design.md)
- [Add opt-in coefficient tests to inference reports](../../../report-review/tasks/coefficient-tests/README.md)

<!-- insitu:end -->
