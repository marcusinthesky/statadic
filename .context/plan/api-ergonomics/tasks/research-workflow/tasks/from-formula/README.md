---
type: task
title: Decline a from_formula constructor
description: "Declined because the documented formula parser already gives a one-line route."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: closed
resolution:
  reason: "formula_to_terms already provides a documented one-line route, so a second constructor adds surface without capability."
---

# Decline a from_formula constructor

## Goal

Proposal: add `PoissonRegression.from_formula(...)` and equivalents. Declined for now.

## Acceptance

- [x] Record the decision and its evidence.

## Why

`docs/reference/composition.md:247` already documents `OLS(*formula_to_terms("y ~ x")).fit(data)`. A second constructor adds API surface without new capability.

## Reopen if

Formula syntax itself needs to express roles such as exposure or offset.

Follow the [handoff guide](../../handoff.md). Measurements are in [evidence](../../evidence.md).

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
