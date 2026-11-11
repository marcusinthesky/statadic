---
type: task
title: Defer confidence bands on fitted curves
description: "Deferred because bands on fitted curves are a new statistical capability needing their own contract and qualification."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/plotspec-extension/README.md"]
status: blocked
---

# Defer confidence bands on fitted curves

## Goal

Record that confidence bands on fitted curves are new statistics, not a plotting feature. STATADIC has no prediction uncertainty today. Do not implement them now.

## Acceptance

- [ ] The owner requests them for a named family.
- [ ] A mathematical contract states the target (mean, rate or link), the method and the reference law.
- [ ] An independent numerical witness exists, such as a statsmodels prediction interface.
- [ ] A prespecified coverage study exists in `src/python/apps/studies`.

## Constraints when unblocked

- `ARCHITECTURE.md` requires new methods to have their own mathematical contracts and qualification before gaining public capabilities.
- Confidence intervals for a mean and prediction intervals for a new observation must stay distinct.
- `plotspec-extension` supplies the ribbon rendering. Only render intervals this work has established.

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
