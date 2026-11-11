# Research-workflow handoff

Read this before starting any sub-task. Each sub-task is written to be implemented
without other context. Its [evidence](evidence.md) records what was measured.

## Settled decisions

Do not reopen these unless the owner asks:

- Library publication is on hold. Do not make the repository public, tag a release,
  mint a DOI, upload a package or submit a paper. The owner-requested manuscript
  draft in `src/latex/projects/01_statadic` executes its listings against the API, so
  an API change updates its listings and chapters and reruns `just latex::generate`.
- `Observations` refuses silent narrowing of explicit 64-bit inputs.
- `PlotSpec` and native PGFplots stay the publication path.
- Publishing workflows belong in the optional `statadic-publishing` package from the
  [accepted design](../../../release-experience/publishing-design.md), not in core STATADIC.
- Renames are clean breaks with no aliases, because nothing has been released.
- Input conveniences cover `statadic.api.regression` models and `PredictionArtifact` only.
  Panel, spatial and survival workflows keep their structured inputs.

Blocked and closed sub-tasks are not implementation work. Blocked ones name their
trigger; closed ones name the condition for reopening them.

## Paths

Line numbers are hints from commit `36168ea`. Confirm them before editing.

| Short form | Full path |
| --- | --- |
| `api/...` | `src/python/packages/statadic/src/statadic/api/...` |
| `discrepancy/...` | `src/python/packages/statadic/src/statadic/discrepancy/...` |
| `geometry/...` | `src/python/packages/statadic/src/statadic/geometry/...` |
| `tests/...` | `src/python/packages/statadic/tests/...` |
| `docs/...` | `src/python/packages/statadic/docs/...` |
| `examples/...` | `src/python/packages/statadic/examples/...` |
| `apps/docs/...` | `src/python/apps/docs/...` |

## Environment

- Run every command through the pinned shell: `devenv shell -- <command>`. Running
  `src/python/.venv/bin/python` outside it fails with a missing `libstdc++.so.6`.
- Read `AGENTS.md`, `src/python/packages/statadic/AGENTS.md` and
  `src/python/packages/statadic/ARCHITECTURE.md` first.
- Run `just setup` once if TeX or locked environments are unprepared. The TeX bundle
  host can return HTTP 429, so retry rather than changing the pinned bundle.

## Source rules

`just python::architecture` enforces these for shipped `statadic` source:

- NumPy, SciPy, `pickle`, `subprocess` and `ctypes` must not be imported.
- `jnp.frombuffer`, `jnp.load`, `jnp.save` and `.to_numpy` are forbidden.
- Imports must not sit inside `try`, and `importlib.util.find_spec` is forbidden.
- Function-local optional imports are allowed. `PlotSpec.to_matplotlib` is the pattern.
- `api` modules live above the scientific owners. Scientific code never imports `api`.

A new module needs a `[[modules]]` entry and an interface entry in
`src/python/tach.toml`, plus a mirrored test file under `tests/`.

## Shared change rules

**Public API.** Update `docs/` and the `[Unreleased]` section of
`src/python/packages/statadic/CHANGELOG.md`. After reviewing the change, run
`just lean::inventory-refresh`. Add new public symbols to the matching host-wrapper
entry in `src/python/apps/studies/catalogue.json`; `just python::studies::check`
reports gaps.

**Artifact schema.** The first sub-task that changes persisted artifact content bumps
`_SCHEMA_VERSION` in `api/artifacts/_store.py` from 2 to 3. Later sub-tasks extend
version 3 without another bump while nothing is released. No migration code is needed.

**Precision tests.** `tests/conftest.py` enables x64 unless `JAX_ENABLE_X64` is set, so
the default test lane cannot see narrowing. Use `with jax.enable_x64(new_val=False):`,
following `tests/api/test_artifacts.py:291`. The docs tests run both precision lanes.

**Scientific scope.** Do not add statistical capabilities under these sub-tasks. A
rendering or input change must not refit, recompute or reinterpret retained values.

## Finishing a sub-task

For the owner-approved swarm, the [execution guide](execution.md) assigns shared
file ownership and stable integration checkpoints. Workers supply focused
verification and independent review findings; the coordinator runs shared
refreshes and full checks, then records verified completion. A passing stable
checkpoint may verify several completed tasks together.

1. Run the focused tests, for example `just python::statadic::test tests/api/test_data.py`.
2. Run `just python::architecture`, `just python::statadic::lint` and
   `just python::statadic::typecheck`.
3. Tick the acceptance criteria you verified and set `status: done` in frontmatter. Put
   longer verification notes in a `verification.md` beside the sub-task's README.
4. Run `just sync`, then `just check`. Report any check that could not run.
5. Do not commit unless the owner asks.

Never edit generated Insitu regions by hand. If a sub-task's premise no longer holds,
stop and report it rather than widening the scope.
