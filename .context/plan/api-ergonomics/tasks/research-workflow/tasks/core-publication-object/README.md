---
type: task
title: Decline a Publication object in core STATADIC
description: "Declined because the accepted design already places publishing workflows in the optional statadic-publishing package."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/release-experience/publishing-design.md"]
status: closed
resolution:
  reason: "The accepted publishing design places these in the optional statadic-publishing package, and the Lean registry already records claim limits."
---

# Decline a Publication object in core STATADIC

## Goal

Proposal: add a `Publication` object, scientific claim manifest and research bundle to core STATADIC. Declined.

## Acceptance

- [x] Record the decision and its evidence.

## Why

The [accepted design](../../../../../release-experience/publishing-design.md) already decides this:

- `statadic-publishing` consumes public report snapshots, and STATADIC never depends on it.
- Its evidence bundle has separate fields for formal compilation, source fidelity, implementation correspondence and statistical qualification.
- `src/lean/registry/claims.json` already records each theorem's assumptions, relationship and limitations.

Useful ideas such as figure provenance should feed that companion's plan.

## Reopen if

The owner amends the accepted design.

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

- [Design how method evidence reaches a manuscript](../../../../../api-cohesion/tasks/report-review/tasks/evidence-design/README.md)

<!-- insitu:end -->
