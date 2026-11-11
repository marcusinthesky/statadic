# Optional research publishing companion

Status: accepted design for a later implementation; no publishing API or
publication automation is shipped by this change.

## Decision

An optional `statadic-publishing` package will consume STATADIC's public report snapshots.
It will live in the Python workspace with the shared lock and depend on STATADIC;
STATADIC will never depend on it. Keep numerical fitting and scientific interpretation
in researcher-authored analyses. Do not serialize live JAX fits or Python pickles
as a replication contract.

The first consumer is a synthetic OLS paper with HC3 inference. Its analysis
generates two report snapshots (fit and inference), a residual diagnostic figure,
one LaTeX table, factual methods metadata and governed citation keys. A manuscript
references generated files through `\input` and `\includegraphics`; researchers
own its prose and conclusions. Reuse the existing Tectonic asset bundle and
reference-record-to-BibTeX synchronization.

## Ownership and interfaces

| Authored input | Owner | Generated output |
| --- | --- | --- |
| Question, estimand, assumptions, claim text | Researcher | Methods metadata and claim-review checklist |
| Data access and transformation code | Researcher | Data identities/hashes and analysis manifest |
| Model, design, inference, seeds | Analysis script | Immutable public report snapshots and diagnostics |
| Citation metadata | Reference records | Existing synchronized BibTeX |
| Manuscript and journal template | Researcher | Tectonic PDF |
| Exact theorem/source mappings | Formal registry | Scoped proof references; pending review stays pending |
| Artifact locations and build graph | Companion project specification | Tables, figures, evidence index and archive |

Use a small Python interface with three operations:

- `build(project: Path, output: Path) -> Bundle`: execute the declared analysis
  script with the locked interpreter, render its outputs and build the manuscript.
- `verify(bundle: Path, project: Path) -> Verification`: check schema version,
  source/dependency/input identities, artifact hashes and required output status.
- `archive(bundle: Path, destination: Path) -> Path`: verify first, then write a
  deterministic archive with fixed member order, normalized timestamps and a
  SHA-256 inventory.

Expose the same operations through a companion CLI, then add repository `just`
wrappers. Do not add commands to STATADIC's core CLI or require Lean for routine fits.
An archive includes source, manifests, lockfiles, redistributable synthetic data,
reports, figures, citations, manuscript and rebuild instructions. External data
are represented by identity, hash and researcher-authored acquisition instructions.

## Evidence bundle v1

Use UTF-8 JSON manifests and ordinary files. The schema has `schema_version: 1`,
bundle/project identifiers, creation time, source revision and dirty status,
source hashes, lockfile hashes, Python and package versions, platform/backend,
precision, random seeds and per-input identity/hash entries.

Each analysis record names its script, estimand, model/design configuration,
inference covariance and reference choices, input identities, assumptions supplied
by the researcher, report snapshots, diagnostics and output files. Define a
companion-owned JSON adapter over the public immutable `Report` fields and
existing `metadata_to_json` conversion. The core currently renders complete
reports as text, HTML and LaTeX, not a full report JSON API. Unavailable values
and warnings must survive the adapter and rendering. Never infer scientific
assumptions merely from an estimator name.

Each output has a stable logical ID, media type, relative path and SHA-256 hash.
Each claim has researcher-authored text, referenced analysis/output IDs, citation
keys, optional registry claim IDs, and an explicit human-review disposition.
Formal compilation, source fidelity, implementation correspondence and statistical
qualification remain separate fields. Their values include pending, unavailable
and not run; absence is never interpreted as approval.

## Data flow and invalidation

`Authored project → analysis → reports/diagnostics → tables/figures/methods metadata
→ manuscript → verified bundle → archive`.

Data, transformations, model/inference options, seeds, code and lockfiles all
participate in the analysis input fingerprint. A changed fingerprint invalidates
dependent outputs. Changed report values invalidate linked claim reviews; the
companion emits a review-required list rather than rewriting conclusions.
Changed formal mappings invalidate their linked mathematical evidence. Do not
carry forward qualification from another configuration without an explicit,
reviewed compatibility rule; v1 always rebuilds.

Missing files, hashes that disagree, path traversal, unknown schema versions,
analysis failure, unavailable required outputs and unresolved citations fail the
build or verification with a retained stage log. Build into a fresh output
directory and write the successful manifest last. Keep partial outputs clearly
failed and never overwrite an accepted archive.

## Implementation tasks and acceptance

1. Add the optional package and versioned JSON schema; validate bundles and report
   adapters with round-trip fixtures preserving unavailable values and warnings.
2. Add the synthetic OLS vertical slice, using the same HC3 report for both JSON
   and LaTeX. Numerically compare its retained values with independent witnesses;
   build the manuscript offline after explicit dependency preparation.
3. Add build, verify and deterministic archive operations. Rebuild the archive in
   a fresh directory without importing from the original checkout.
4. Demonstrate a reviewer revision from HC3 to another supported covariance:
   coefficients stay fixed, uncertainty/table artifacts change and linked claims
   require human review. The original accepted bundle remains inspectable.
5. Exercise changed data, seeds, lockfiles and formal mappings; failed analyses;
   missing citations; tampered artifacts; unsupported schema versions; unavailable
   inference; and clean reconstruction from synthetic data.

Paper-wide claim tracking, generic analysis scheduling, journal integrations,
external manuscript services and new theorem authoring are outside this first
vertical slice. They require later design decisions grounded in its users.

## Amendment (2026-11-06)

The accepted decision stands. Work landed after acceptance changes how the
first slice is built:

- STATADIC now persists complete reports. `save_analysis` writes schema-3 JSON
  holding parameters (including unavailable values), metrics, metadata,
  warnings, notes and plots, and `load_analysis` restores them. The companion
  references these snapshots as outputs and defines no separate report adapter.
- `statadic-evidence` already provides canonical JSON, hashes, inventories and
  deterministic archive writing and extraction. Build on it rather than adding
  another archive format.
- `src/latex` pins Tectonic to a vendored 6.2 MB bundle and checks logs with
  `tools/check-log`. The companion compiles with that bundle and includes it in
  archives, so a rebuild needs no network.
- The residual diagnostic figure comes from `statadic.api.plots.residual_plot`,
  added by the research-release `walkthroughs` task, rather than from
  hand-built plot coordinates.

The [companion handoff](../publishing-companion/handoff.md) gives the package
wiring and task specifications.
