---
type: task
title: Implement build verify and deterministic archive
description: "build, verify and archive over a publishing project, with stage logs, a manifest written last and a deterministic archive."
links: [".context/plan/release-experience/publishing-design.md", ".context/plan/publishing-companion/handoff.md"]
status: done
---

# Implement build verify and deterministic archive

## Goal

Implement the three operations from the design over a `publishing.toml` project, and expose them as a CLI and `just` recipes.

Depends on [bundle](../bundle/README.md) and [paper](../paper/README.md).

## Acceptance

- [x] `statadic_publishing.build(project: Path, output: Path) -> Bundle`, `verify(bundle: Path, project: Path) -> Verification` and `archive(bundle: Path, destination: Path) -> Path` exist.
- [x] `python -m statadic_publishing build|verify|archive` exposes them.
- [x] `just python::publishing::build`, `::verify` and `::archive` wrap the CLI.
- [x] Building the `ols-hc3` example, verifying it and archiving it succeeds.
- [x] Archiving the same bundle twice yields identical bytes.
- [x] Tests cover each failure listed below.
- [x] `just python::publishing::check` and `just check` pass.

## Specification

**`build`**:

1. Refuse an `output` directory that exists and is not empty.
2. Stage the project sources and the declared external files into the bundle.
3. Check each input's sha256 against the specification.
4. Run the analysis with `sys.executable -I` and a minimal environment that keeps the JAX x64 setting.
5. Check that every required output exists.
6. Compile the manuscript with `tectonic -X compile --untrusted --only-cached --bundle <staged zip>`, then run `check-log`.
7. Write the manifest last.

Each stage writes `logs/<stage>.log`. On failure, write `FAILED` with the stage name and keep the logs and partial outputs. Write no manifest, and never touch an existing accepted bundle.

**`verify`**:

- Check the schema version.
- Run `statadic_evidence.verify_inventory`.
- Recompute the source, lockfile and input hashes against the project.
- Confirm every required output is `available`.
- Return a `Verification` that lists each failure, rather than stopping at the first.

**`archive`**:

- Run `verify` first.
- Refuse an existing destination.
- Call `statadic_evidence.write_archive`.
- The archive holds the staged TeX bundle zip, about 6.2 MB, so a rebuild needs no network.
- It also holds a `REBUILD.md` naming the packages and versions to install, and the commands to run.

**Failure tests**:

- a tampered output;
- a stale input whose hash changed;
- an unknown schema version;
- a path that traverses out of the bundle;
- an analysis that raises, which must keep its log and write no manifest;
- an existing destination;
- an unresolved citation, which must fail at the manuscript stage.

## Non-goals

- No incremental rebuilds; v1 always rebuilds.
- No scheduler.
- No upload.

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
