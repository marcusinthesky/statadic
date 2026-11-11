---
type: task
title: Defer a fitted-result plot accessor
description: "Deferred until the walkthroughs show that the plot helpers read poorly in practice."
links: [".context/plan/research-release/handoff.md", ".context/plan/research-release/evidence.md", ".context/plan/research-release/tasks/walkthroughs/README.md"]
status: blocked
---

# Defer a fitted-result plot accessor

## Goal

Record the proposed `fit.plot.<kind>(...)` convenience, such as `fit.plot.rate("load", grid)`. Do not implement it now. The question is discoverability, not rendering infrastructure. The `walkthroughs` task supplies the evidence for that question.

## Acceptance

- [ ] [walkthroughs](../walkthroughs/README.md) is done, and its `verification.md` records concrete friction with `prediction_line`, `coefficient_plot` or `residual_plot`.
- [ ] The owner asks for the accessor.
- [ ] A design names each accessor method and the existing helper it calls.

## Constraints when unblocked

- Each method delegates to a `statadic.api.plots` helper or to `Report.coefficient_plot`. It computes nothing new.
- No accessor hierarchy across families until one family's accessor has been used.
- Keep the module-level helpers. They are the documented path, and the accessor would only add aliases.

Follow the [handoff guide](../../handoff.md). Findings are in [evidence](../../evidence.md).

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
