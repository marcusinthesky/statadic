---
type: task
title: Repair numerical boundaries and evidence integrity
description: Repair numerical boundaries and evidence integrity
links: []
status: in-progress
---

# Repair numerical boundaries and evidence integrity

## Goal

Repair the eleven confirmed findings at `937f390` before extending statistical
capabilities. Three GPT-6.1 Terra medium workers own disjoint changes; the
coordinator owns integration and evidence. Five additional review findings have
not been supplied and are not claimed resolved.

## Acceptance

- [x] Preserve panel identities and integer MMD arithmetic.
- [x] Preserve artifact precision and canonical inactive geometry.
- [x] Make padded exact transport agree across eager/JAX paths.
- [x] Restrict paired summaries to explicit scientific comparison groups.
- [x] Correct the IV description without changing its generator.
- [x] Match compiler policy and protect immutable performance inputs.
- [ ] Provision release reference dependencies and diagnose the cold build.
- [x] Pass affected focused checks and record hosted validation limitations.

## Remaining merge gate

The release workflow now provisions the locked reference group before offline
preflight. The pinned layout-tool wheel builds in a fresh local cache, but that
does not explain the recorded hosted exits 245/249. No speculative pin change
or enforcement bypass was introduced. A cold x86-64/ARM64 hosted run remains
necessary; this task stays in progress until that evidence exists.

Focused verification and review limits are recorded in the
[integration record](../phase-4/tasks/verification/README.md).

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
