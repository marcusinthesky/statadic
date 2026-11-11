---
type: task
title: Document and check the saved-artifact compatibility contract
description: "State what predictors, analysis snapshots and reproduction each promise, record the writing version, and check the contract on the installed wheel."
links: [".context/plan/research-release/handoff.md", ".context/plan/research-release/evidence.md"]
status: done
---

# Document and check the saved-artifact compatibility contract

## Goal

Before any release, settle and document what STATADIC promises about saved artifacts across versions. Separate three things: saving a predictor, saving a rendering-only analysis snapshot, and preserving enough to reproduce inference. Then check the promise against the installed wheel. Orbax stays out of scope.

## Acceptance

- [x] `docs/reference/artifacts.md` exists, is linked from `docs/reference/index.md`, and states the three tiers and the version rule below.
- [x] Each manifest records its writer, and the schema rejection message names the version found and the version supported.
- [x] The new tests below pass, and the existing corruption tests in `tests/api/test_artifacts.py` are unchanged.
- [x] The installed-wheel check covers the cases below.
- [x] The CHANGELOG `[Unreleased]` section and `docs/reference/composition.md` point to the new page.
- [x] Focused tests, `just python::statadic::test-artifacts`, `just python::architecture`, lint, typecheck and `just check` pass.

## Specification

**Three tiers**, documented in `docs/reference/artifacts.md`:

| Tier | Written by | Restores | Never restores |
| --- | --- | --- | --- |
| Predictor | `save_predictor`, `save_embedding_predictor` | Prediction or transformation with the declared roles | Inference, covariance or refitting |
| Analysis snapshot | `save_analysis` | Report values, warnings, notes and plots for rendering; TeX markup only with `trust_tex=True` | A fit or sampling uncertainty |
| Reproduction | Not STATADIC; planned for `statadic-publishing` | Source, lockfile, data identity and rebuild instructions | Not applicable |

**Version rule** before a public release:

- A reader accepts only its own schema version, and any change to persisted content bumps that version.
- There is no migration. A user re-saves from the original analysis.
- The writer version is recorded for diagnosis only and is never compared.

The page says the post-release promise needs the owner's confirmation in `publication-hold`.

**Code** in `api/artifacts/_store.py`:

- `write_bundle` adds `"writer": {"package": "statadic", "version": ...}`, taking the version from the same source as `statadic.__version__`. Avoid a circular import.
- `read_bundle` adds `writer` to the required manifest keys, checks that it is a record of two strings, and does not compare versions.
- The schema message becomes, for example, `Artifact schema version 2 is not supported; statadic 0.x reads schema 3 only. Re-save it from the original analysis.` It must not fail when the found value is not an integer.
- This extends schema 3 without a bump.

**Tests** in `tests/api/test_artifacts.py`:

- the writer is recorded;
- an artifact with a different writer version still loads;
- a schema-2 manifest raises a message that names 2 and 3;
- a missing `writer` key is rejected.

**Installed-wheel check:** extend `_INSTALLED_CHECK` in `scripts/check_distribution.py`. Add no new runtime dependencies; build inputs with `Observations.from_arrays`. Cover:

- Poisson and logistic predictors: live and loaded predictions agree for every supported `kind`;
- a corrupted manifest and a schema-2 manifest are each rejected with `ArtifactError`;
- an artifact saved with x64 arrays, loaded in a fresh `JAX_ENABLE_X64=0` subprocess, is refused rather than narrowed;
- `load_analysis` refuses a snapshot containing TeX markup unless `trust_tex=True`.

## Non-goals

- No Orbax.
- No migration code.
- No cross-version loading.
- No change to what any tier restores.

Follow the [handoff guide](../../handoff.md). Findings are in [evidence](../../evidence.md).

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
