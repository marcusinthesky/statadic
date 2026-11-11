---
type: task
title: Retain deferred quantities and scientific extensions
description: Retain remaining quantity workflows and scientifically distinct extensions outside this alpha repair slice.
links: []
status: backlog
---

# Retain deferred quantities and scientific extensions

## Goal

Retain remaining quantity workflows and scientifically distinct extensions outside this alpha repair slice.

Owner: Backlog. No prerequisite implementation.
See the parent [design](../../design.md), [dispositions](../../proposal-dispositions.md)
and [validation](../../validation.md). Numerical test execution belongs to the coordinator.

See [deferred work and reopening criteria](deferred.md).

## Acceptance

- [ ] Adjudicate each follow-up against an explicit use case and scientific contract.
- [ ] Implement separately qualified quantity adapters and extensions only after their recorded triggers.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| backlog | [Design directional coefficient tests](tasks/directional-tests/README.md) | Specify a one-sided test for a single linear restriction, or document the alternative slot as reserved. |
| backlog | [Make the NB2 log likelihood accurate in float64](tasks/nb2-log-likelihood/README.md) | Replace the single-precision-grade log-beta in the NB2 row likelihood without losing small-dispersion or float32 accuracy. |
| backlog | [Decide repository hygiene before publication](tasks/publication-hygiene/README.md) | Owner decisions on private paths, history, scans and internal identifiers before the repository becomes public. |
| backlog | [Compare shipped unreferenced variants with installed peers](tasks/reference-gaps/README.md) | Add matched comparisons for shipped variants that lack one, and record a reason where no peer applies. |
| backlog | [Test invariants across Regression routes](tasks/route-invariants/README.md) | Test constructor compatibility, prepared-state ownership, retained schema, result type and no silent model substitution across every supported Regression route. |
| backlog | [Account for standalone inferential procedures](tasks/standalone-procedures/README.md) | Record a host decision for every inferential procedure outside the five workflow APIs and state that set in the orientation. |

`6 open; 1 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
