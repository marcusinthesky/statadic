---
type: task
title: Complete family inference and panel workflows
description: Reuse host inference reports and align supported panel vocabulary.
links: []
status: done
---

# Complete family inference and panel workflows

## Goal

Complete the Phase 2 host API changes with the existing statistical components,
preserving native evidence and explicit reference-law choices.

## Acceptance

- [x] Preserve fixed constructor conventions and prepared-instance ownership.
- [x] Add GMM and GEL host reports with native state and availability reasons.
- [x] Align panel/effect spelling and prediction kinds without changing estimators.
- [x] Preserve spatial study IDs and record the separate SAC stopping issue.
- [x] Refresh metadata and pass the complete repository check.

## Implementation

GMM uses the existing generic `InferenceResult` and a family sandwich descriptor;
GEL uses one host result around native test output. Numerical pytrees have no
new reporting responsibilities. Panel presets map to the same `Within` flags;
spatial `unit` replaces the previous `fixed` option. No new persistence codec,
universal specification or inference defaults were introduced.

The sibling spatial-panel-stopping task retains the seed-0 numerical observation.
Profile-equality properties and the original seed-7 optimizer parity test remain
separate, with their tolerances preserved.

## Verification

On 2026-11-05, `devenv shell -- just check` passed completely, including:

- Package: 2,549 passed, 166 skipped; strict structural typing and lint passed.
- Required shape lane: 7 passed and no static errors.
- Evidence: 15 passed; studies: 287 passed, 3 skipped; catalogue check passed.
- Performance: 24 passed.
- Docs: 83 float32 tests and 16 selected x64 tests passed; site/labs built.
- Lean certification, blueprint, registry, LaTeX and agent checks passed.

The final full log is `/tmp/statadic-phase2-final-check.log`. The first full attempt
exposed the random-effects SEM boundary witness retained in the sibling task;
the fitted reference now retains its original seed-38 fixture and tolerances.
These source checks do not establish retained statistical qualification.

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
