---
type: task
title: Add covariance and moment estimands
description: Add one host Estimand for covariance and supplied moments over existing scientific owners.
links:
  - .context/plan/api-cohesion/tasks/estimand/implementation.md
status: done
---

# Add covariance and moment estimands

## Goal

Add one host Estimand for covariance and supplied moments over existing scientific owners.

Owner: Agent B. Depends on [contracts](../contracts/README.md), [lifecycle-typing](../lifecycle-typing/README.md).
See the parent [design](../../design.md), [dispositions](../../proposal-dispositions.md)
and [validation](../../validation.md). Numerical test execution belongs to the coordinator.

See [implementation evidence and pending checks](implementation.md).

See [implementation evidence](implementation.md).

## Acceptance

- [x] Implement explicit quantity/estimator composition with one owned data source.
- [x] Move covariance bricks and preserve covariance/GMM/GEL evidence.
- [x] Retire competing host estimation doors and verify aligned state ownership.

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

- [Type contrast estimands and rename the core estimand root](../final-review/tasks/contrast-estimand/README.md)

<!-- insitu:end -->
