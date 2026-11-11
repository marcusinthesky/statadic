---
type: task
title: Review the whole stack for a coherent alpha
description: Whole-stack design, scientific evidence, JAX, performance and simplification review; repairs remain separate.
links: [".context/plan/research-release/tasks/alpha-review/brief.md", ".context/plan/research-release/tasks/alpha-review/decisions.md", ".context/plan/research-release/tasks/alpha-review/coverage.md", ".context/plan/research-release/tasks/alpha-review/findings.md", ".context/plan/research-release/tasks/alpha-review/verification.md", ".context/plan/research-release/tasks/alpha-review/backlog.md"]
status: done
---

# Review the whole stack for a coherent alpha

## Goal

Review candidate `17fdc54` against `52268aa` and the accepted chat decisions.
The coordinator owns cohesion and alpha dispositions. Leaf reviewers inspect
bounded scientific slices through independent lenses. Review completion and
release readiness are separate. Publication remains on hold.

Start with [the brief](brief.md), [decisions](decisions.md) and
[coverage](coverage.md). Findings and verification never imply unexecuted checks.
The [design adjudication](synthesis.md) recommends holding this candidate;
[the backlog](backlog.md) contains sixteen repair workstreams and thirty-three
cleanup dispositions. All 1,433 changed paths have bounded inspection records.

## Acceptance

- [x] Freeze the stack, intent ledger and operation/evidence baseline.
- [x] Account for every changed path and scientific operation with honest review status.
- [x] Complete the scientific and execution/simplification passes for every slice.
- [x] Adjudicate findings, independent challenges and alpha dispositions.
- [x] Order concrete repairs and frozen-candidate verification requirements.
- [x] Run one coordinated `just check` and record failed or unexecuted stages.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 9 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
