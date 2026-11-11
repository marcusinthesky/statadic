---
type: task
title: Defer pandas ExtensionArray and xarray integration
description: "Deferred until a labelled-output or measure-per-cell use case exists; Arrow input already covers ordinary pandas frames."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/table-inputs/README.md"]
status: blocked
---

# Defer pandas ExtensionArray and xarray integration

## Goal

Record the deferred pandas `ExtensionArray` and xarray ideas. Neither is needed for ordinary tabular input once `table-inputs` lands. Do not implement them now.

## Acceptance

- [ ] The owner names a concrete use case below.
- [ ] A design states the optional dependency, alignment policy and JAX transformation boundary.

## Triggers

- **xarray:** labelled outputs with several axes, such as survival curves by subject and time.
- **`ExtensionArray`:** pandas users need one empirical distribution per cell, backed by `PackedClouds`.

## Constraints when unblocked

- Stay optional and outside the JAX core.
- Never inherit silent joins, drops or fills; require exact alignment or an explicit policy.
- `xarray_jax` treats standard coordinates as static under JIT, so measure recompilation first.

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
