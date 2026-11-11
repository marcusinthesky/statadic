---
type: task
title: Add the project specification and evidence bundle schema v1
description: "Create the statadic-publishing package with a strict publishing.toml reader and evidence manifest v1, reusing STATADIC analysis snapshots for report JSON."
links: [".context/plan/release-experience/publishing-design.md", ".context/plan/publishing-companion/handoff.md"]
status: done
---

# Add the project specification and evidence bundle schema v1

## Goal

Create the `statadic-publishing` package and its two data contracts: the authored project specification and the generated evidence manifest. The design called for a separate "report adapter". `save_analysis` snapshots now serve as report JSON, so this task only references them.

## Acceptance

- [x] The package is wired as the [handoff](../../handoff.md) describes, and `just python::architecture` and `just python::publishing::check` pass.
- [x] `statadic_publishing.project` reads `publishing.toml` strictly into frozen dataclasses.
- [x] `statadic_publishing.manifest` defines manifest v1 as frozen dataclasses, with strict JSON reading and writing through `statadic_evidence.write_json`.
- [x] Tests reject an unknown schema version, unknown or missing keys, a missing status, an unknown status value, absolute paths and `..` traversal.
- [x] A test bundle holding a `save_analysis` snapshot with None values, warnings and notes loads back through `load_analysis` unchanged. Assumptions and review statuses round-trip through the manifest.

## Specification

**`publishing.toml`**, with all paths relative to the project directory:

- `[project]`: `id` and `title`.
- `[analysis]`:
  - `script`;
  - `seed`, an integer;
  - `estimand`, as text;
  - `assumptions`, a list of researcher-authored text;
  - `model`, as a table of factual configuration;
  - `covariance` and `reference`.
- `[[inputs]]`:
  - `id`;
  - either `path` or `identity` for external data;
  - `sha256`;
  - `redistributable`;
  - `acquisition`, required when the input is not redistributable.
- `[[outputs]]`: `id`, `path` under the generated directory, `media_type` and `required`.
- `[manuscript]`:
  - `root` and `main`;
  - `bibliography` and `tex_bundle`, each with its `sha256`;
  - the path to `check-log`.
- `[environment]`: `lockfiles`, as a list.
- `[[claims]]`:
  - `id` and `text`;
  - `outputs` and `citations`;
  - `registry_claims`, optionally;
  - the statuses `formal`, `source_fidelity`, `implementation`, `qualification` and `human_review`.

Paths may leave the project directory only for `bibliography`, `tex_bundle`, `check-log` and `lockfiles`. The build copies those files into the bundle under fixed names.

**Status values** are `pending`, `unavailable`, `not-run`, `passed`, `failed` and `approved`. `human_review` accepts only `pending` or `approved`. A missing status is an error and is never read as approval.

**Manifest v1**, written last by `build`, contains:

- `schema_version: 1`, the bundle and project ids, and the creation time in UTC;
- the source revision, with its dirty state, or `unavailable` outside a Git checkout;
- source, lockfile and input hashes;
- the Python, `statadic`, `jax` and `statadic-publishing` versions;
- the platform, the JAX backend, the x64 setting and the seed;
- one record per analysis: script, estimand, model, covariance, reference, input ids, assumptions, snapshot output ids and diagnostic output ids;
- the outputs, each with id, media type, relative path, sha256 and `status` (`available` or `unavailable`);
- the claims, copied from the specification.

The manifest never infers an assumption or status from an estimator name.

## Non-goals

- No build, verify or archive operations; [operations](../operations/README.md) owns them.
- No paper-wide claim graph.
- No changes to STATADIC's report JSON.

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
