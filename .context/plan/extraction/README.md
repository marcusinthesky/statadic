---
type: plan
title: Extract STATADIC into a standalone repository
description: Extract STATADIC into a standalone repository
links: []
status: done
---

# Extract STATADIC into a standalone repository

## Goal

Extract STATADIC into a standalone repository

## Acceptance

- [X] Copy the existing STATADIC package without changing its scientific code.
- [X] Replace example dependencies and regenerate the standalone workspace lock.
- [X] Pass the complete package fast suite and distribution checks.
- [X] Pass repository checks after synchronizing generated views.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Provenance

Copied 452 package files byte-for-byte from Pricing Perspective worktree
`89a59b400fdba1280ed7d35753e6c2085cdd8136`. Removed the template library and
its sample consumer. Package code, tests, notebooks and configuration are retained.
The standalone workspace lock was derived from the source workspace lock,
preserving qualified numerical and development versions, including JAX 0.11.0,
NumPy 2.4.6, SciPy 1.18.0 and pytest 8.4.2.

## Verification

- Full `just check`: passed; complete fast suite 2,677 passed, 161 skipped.
- `just python::statadic::test-artifacts`: both wheel routes and standalone sdist
  development passed, including 45 extracted-sdist typing tests.
- All 452 copied package files match the source byte-for-byte.
- Optional external-reference, slow, GPU and qualification suites were not run.

Initial fresh dependency resolution selected newer numerical versions and
produced three fast-suite failures. Retaining the source lock resolved all three.
The template pytest override was removed so package tooling remains authoritative.

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
