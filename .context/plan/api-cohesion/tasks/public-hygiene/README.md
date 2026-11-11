---
type: task
title: Retire dead public code and generated clutter
description: Remove proven clutter and obsolete public code while retaining scientific and historical evidence.
links: []
status: done
---

# Retire dead public code and generated clutter

## Goal

Remove proven clutter and obsolete public code while retaining scientific and historical evidence.

Owner: Coordinator. Depends on [contracts](../contracts/README.md).
See the parent [design](../../design.md), [dispositions](../../proposal-dispositions.md)
and [validation](../../validation.md). Numerical test execution belongs to the coordinator.

## Acceptance

- [x] Remove the orphaned cache and consumer-proven unused exports.
- [x] Migrate internal row IDs and verify replacement spectrum consumers.
- [x] Record owner decision to retain embedding until actual downstream migration.
- [x] Refresh current prose and paths without deleting linked review evidence.

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
