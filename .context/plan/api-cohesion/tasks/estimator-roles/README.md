---
type: task
title: Move structural roles into numerical problems
description: Make model roles explicit on numerical problems while preserving existing algorithms and precision.
links:
  - .context/plan/api-cohesion/tasks/estimator-roles/implementation.md
status: done
---

# Move structural roles into numerical problems

## Goal

Make model roles explicit on numerical problems while preserving existing algorithms and precision.

Owner: Agent A. Depends on [contracts](../contracts/README.md).
See the parent [design](../../design.md), [dispositions](../../proposal-dispositions.md)
and [validation](../../validation.md). Numerical test execution belongs to the coordinator.

See [implementation evidence](implementation.md).

## Acceptance

- [x] Move spatial process/effects/lag choices and Within effects into problems.
- [x] Migrate or reject structural roles for every shared estimator consumer.
- [x] Preserve independent numerical, pytree and compilation contracts.

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

- [Estimator role migration implementation](implementation.md)

<!-- insitu:end -->
