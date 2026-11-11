---
type: task
title: Validate failures and independent reconstruction
description: "Rebuild the synthetic paper from its archive in a fresh environment outside the checkout, and exercise the required failures end to end."
links: [".context/plan/release-experience/publishing-design.md", ".context/plan/publishing-companion/handoff.md"]
status: done
---

# Validate failures and independent reconstruction

## Goal

Show that the archive is a self-sufficient replication artifact: someone holding only the archive and the named package versions can rebuild the paper and get the same numbers.

Depends on [operations](../operations/README.md).

## Acceptance

- [x] An end-to-end test rebuilds the archive outside the original checkout:
  1. build fresh `statadic` and `statadic-publishing` wheels;
  2. install them into a new venv outside the checkout;
  3. extract the archive with `statadic_evidence.extract_archive` into an empty directory;
  4. run `python -I -m statadic_publishing build` on the extracted project into a new output directory;
  5. `verify` the result.
- [x] Generated TeX, JSON and snapshot files are byte-identical to the original bundle's. The PDF only needs to compile and pass `check-log`.
- [x] End-to-end failures are exercised:
  - changed data fails `verify`;
  - a missing citation fails the manuscript stage with its log;
  - an analysis that raises leaves logs and no manifest;
  - an unavailable required output fails the build;
  - a declared non-required output that the analysis does not write is recorded as `status: unavailable`, and the build still succeeds;
  - an inference snapshot whose rows are unavailable keeps them as None through build, archive and rebuild.
- [x] The test runs in `just python::publishing::check`, or is marked slow and documented in the package README with the command that runs it.
- [x] `verification.md` records the rebuild commands and the hashes compared.

## Non-goals

- No cross-platform byte identity for PDFs.
- No reviewer revision; [revision](../revision/README.md) owns it.

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

Follow the [handoff](../../handoff.md) and the [accepted design](../../../release-experience/publishing-design.md), including its 2026-11-06 amendment.
