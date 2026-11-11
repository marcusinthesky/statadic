---
type: task
title: Correct the advertised inference guarantees
description: "Remove the familywise, non-finite p-value and equivalence guarantees that fail, and check every new source, before any proof work is presented."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md"]
status: done
---

# Correct the advertised inference guarantees

## Goal

Make STATADIC ship no false statistical promise. Three code lanes run in parallel. A sources lane then independently checks every citation they added, and fixes the export that drops DOIs.

Each change removes or narrows a guarantee. None adds a scientific procedure.

## Acceptance

- [x] [multiplicity](tasks/multiplicity/README.md), [nonfinite-pvalues](tasks/nonfinite-pvalues/README.md), [equivalence](tasks/equivalence/README.md) and [sources](tasks/sources/README.md) are complete.
- [x] Each code lane was cross-reviewed by another code lane before integration.
- [x] The coordinator applied the lanes' requests:
  - inventory refresh;
  - catalogue variants;
  - CHANGELOG entries;
  - `method-catalogue.md`, `package-development.md` and `known-limitations.md`.
- [x] `just python::studies::check` passes at the integration checkpoint.

Integration check, 2026-11-07: the lane requests are applied (catalogue,
CHANGELOG, `package-development.md`, `known-limitations.md`). The equivalence
wording request for `equivalence.py` and `mixture/contracts.py` was applied
that day. `just python::studies::check` passes after updating the exact DISCO
test to distinct arrangements; `just latex::check` and the docs check pass.

## Order

| Wave | Work | Owns |
| --- | --- | --- |
| 1 | [multiplicity](tasks/multiplicity/README.md) | `multiplicity.py`, its tests and its studies adapters |
| 1 | [nonfinite-pvalues](tasks/nonfinite-pvalues/README.md) | The permutation kernel and three other add-one producers |
| 1 | [equivalence](tasks/equivalence/README.md) | `equivalence.py` and the mixture contract docstrings |
| 2 | [sources](tasks/sources/README.md) | Checking every new reference record, and the DOI export |

Cross-review runs multiplicity → nonfinite-pvalues → equivalence → multiplicity.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 4 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
