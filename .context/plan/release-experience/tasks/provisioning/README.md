---
type: task
title: Repair formatting and reproducible validation
description: Repair formatting and reproducible validation
links: []
status: in-progress
---

# Repair formatting and reproducible validation

## Goal

Repair formatting and reproducible validation

## Acceptance

- [X] Separate formatters and exercise them in a disposable checkout.
- [X] Build Lean with direct extraction and expose pinned native libraries.
- [X] Add guarded hosted-runner cleanup, capacity logs and artifact retention.
- [ ] Observe cold-cache and warm-cache hosted checks on x86-64 and ARM64.

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

## Implementation

Local toolchain identity, capacity and formatter results are retained under `build/ci/`. Hosted execution requires the changed workflow to run.
