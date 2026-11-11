---
type: plan
title: Student-friendly documentation and interactive labs
description: Student-friendly documentation and interactive labs
links: []
status: done
---

# Student-friendly documentation and interactive labs

## Goal

Deliver a Sphinx/MyST learning site with three executable tutorials and six native marimo labs, preserving STATADIC scientific behavior and existing advanced guides.

## Acceptance

- [X] Publish a beginner learning path, tutorials, reference and retained advanced guides.
- [X] Execute all six native labs with static alternatives and meaningful numerical checks.
- [X] Integrate a locked docs app and strict build with repository checks.
- [X] Complete the prescribed repository gates and artifact verification; report limitations.

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

## Decisions

Sphinx/MyST with PyData theme; canonical content stays in the STATADIC package docs.
The internal docs app owns tools and labs without adding runtime dependencies.
Three GPT-6.1 Terra agents at medium reasoning effort own content, platform and labs.
No scientific API, proof, hosting or release changes are included.

## Verification

See [verification evidence and limits](verification.md) for commands, results,
the corrected module path and checks not run.
