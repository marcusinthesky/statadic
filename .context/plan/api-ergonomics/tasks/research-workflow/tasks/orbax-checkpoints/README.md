---
type: task
title: Decline Orbax checkpointing for now
description: "Declined for now because STATADIC persists small declarative predictors and has no training state to checkpoint."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/release-experience/publishing-design.md"]
status: closed
resolution:
  reason: "STATADIC has no checkpointable training state, typical artifacts are tiny, and Orbax adds 58 packages for no current use case."
---

# Decline Orbax checkpointing for now

## Goal

Proposal: add Orbax-backed checkpoints and a public checkpointable pytree. Declined for now.

## Acceptance

- [x] Record the decision and its evidence.

## Why

- A typical regression artifact is 8 bytes, and STATADIC fits have no long-running training state to resume.
- The accepted publishing design says not to serialize live JAX fits as a replication contract.
- Orbax adds 58 packages. Unconstrained it selects JAX 0.12, which `pyproject.toml` excludes.
- Reading 5,000,000 elements takes about 1.5 s, which does not justify this for current artifacts.

## Reopen if

A retained workflow needs resumable iterative state or sharded multi-device arrays, or artifact read time becomes a measured problem. First consider a faster stdlib decoder; `jnp.frombuffer` is forbidden by the architecture guard.

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
