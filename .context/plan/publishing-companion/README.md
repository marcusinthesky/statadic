---
type: plan
title: Optional reproducible research publishing companion
description: "The optional statadic-publishing package: build, verify and archive one reproducible publication bundle. First slice is the synthetic OLS/HC3 paper."
links: [".context/plan/release-experience/publishing-design.md", ".context/plan/publishing-companion/handoff.md", ".context/plan/research-release/README.md"]
status: in-progress
---

# Optional reproducible research publishing companion

## Goal

Tie analysis source, data identities, STATADIC reports and figures, environment and claim statuses into one verified bundle that rebuilds from its archive. This makes a publication-native workflow real rather than promised.

## Acceptance

- [ ] Deliver the optional companion described by the accepted design.
- [x] Rebuild the synthetic manuscript and evidence from an independent archive.

## Start here

Read the [handoff](handoff.md). The first slice is `bundle`, `paper`, `operations` and `replication`, in the order the [research-release plan](../research-release/README.md) gives. `revision` follows once that slice is done. Publication stays on hold.

The first slice passed the [integrated checkpoints](../research-release/verification.md).
The later reviewer-revision slice remains in backlog and was not implemented.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| backlog | [Demonstrate inference revision and claim invalidation](tasks/revision/README.md) | Demonstrate inference revision and claim invalidation |

`1 open; 4 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Hold the software paper and public release](../api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md)
- [Specify the optional publishing companion](../release-experience/tasks/publishing-design/README.md)
- [Prove one coherent research workflow before release](../research-release/README.md)

<!-- insitu:end -->

## Implementation

Follow the [handoff](handoff.md) and the [accepted design](../release-experience/publishing-design.md), including its 2026-11-06 amendment.
