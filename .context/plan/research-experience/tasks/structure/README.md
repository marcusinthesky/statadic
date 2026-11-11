---
type: task
title: Audit module organization and test ownership
description: Audit module organization and test ownership
links: []
status: done
---

# Audit module organization and test ownership

## Goal

Identify filename-encoded hierarchies, mixed responsibilities and inconsistent
scientific ownership across runtime code, applications, proofs and pytest.
Recommend concrete moves without treating every flat directory as defective.
The broader implementation programme continues; runtime moves remain proposals.

## Acceptance

- [x] Inspect runtime modules, application boundaries and direct pytest imports.
- [x] Propose role/family packages with concrete source and test destinations.
- [x] Document intentional exceptions and migration/verification requirements.
- [x] Synchronize repository views and run repository checks.

[Audit and migration order](audit.md) · [Runtime and pytest detail](runtime.md) ·
[Static inventory](inventory.json) · [Verification](verification.md)

The inventory covers 220 runtime Python files and 186 pytest modules. Direct
imports identify candidate ownership, not executed test coverage. No runtime
module or test was moved by this audit. Studies' separately authorized package
refactor is implemented and tested.

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
