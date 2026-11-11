---
type: task
title: Unify spatial and survival workflows
description: Unify live spatial inputs, names and prediction while preserving graph and survival semantics.
links:
  - .context/plan/api-cohesion/tasks/spatial-survival/implementation.md
status: done
---

# Unify spatial and survival workflows

## Goal

Unify live spatial inputs, names and prediction while preserving graph and survival semantics.

Owner: Agent A. Depends on [estimator-roles](../estimator-roles/README.md), [lifecycle-typing](../lifecycle-typing/README.md).
See the parent [design](../../design.md), [dispositions](../../proposal-dispositions.md)
and [validation](../../validation.md). Numerical test execution belongs to the coordinator.

See [implementation evidence](implementation.md).

## Acceptance

- [x] Accept canonical spatial inputs and share period/node prediction state.
- [x] Apply graph-qualified names and the no-explicit-intercept fixed-effects rule.
- [x] Bind replayable survival draws in SpatialSurvivalData; default to median prediction.
- [x] Preserve explicit refinement, native graph learning limits and useful errors.

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

- [Spatial survival lifecycle implementation](implementation.md)

<!-- insitu:end -->
