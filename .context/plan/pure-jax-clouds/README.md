---
type: plan
title: Pure JAX and canonical empirical clouds
description: Pure JAX and canonical empirical clouds
links: []
status: in-progress
---

# Pure JAX and canonical empirical clouds

## Goal

Pure JAX and canonical empirical clouds

## Acceptance

- [x] Consolidate empirical distributions into one transform-safe PackedClouds.
- [x] Port shipped NumPy/SciPy uses and enforce source purity.
- [x] Replace smooth handwritten derivatives and qualify exceptional rules.
- [ ] Validate numerical, typing, artifact and architecture contracts; run just check.

Implementation scope and mathematical limits are recorded in
[implementation](implementation.md), with checks in [verification](verification.md).
Implementation and scoped validation are complete. Workspace-wide validation
remains open because concurrent prediction, regression and spatial-panel work
changed the checkout during the final checks; see the verification record.

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
