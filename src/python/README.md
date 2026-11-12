---
description: Python workspace and repository scripts.
---

# Python

The `packages/statadic` library contains the statistical implementation, package
tests, notebooks, documentation and distribution checks copied from Pricing
Perspective. Its scientific code and package configuration are preserved.

Run from the repository root:

```bash
just python::setup
just python::run
just python::check
just python::architecture
just python::statadic::test-artifacts
```

The workspace shares one `uv.lock`. STATADIC's own development dependencies and
checker versions govern its lint and type checks. Tests run in the package's own
process and configuration. Locally, `just python::check` runs only the fast
tests Testmon selects as affected; with `CI` set it runs the complete fast suite,
as does `just python::test-fresh`. CI may explicitly set `STATADIC_TEST_MODE` to
`record` to run all fast tests while recording dependencies, or `affected` to
select tests from a compatible Testmon database. `fresh` disables selection and
recording; release checks force it. A missing database makes `affected` run the
full fast suite. Hosted PR checks use `affected`; pushes to `main` use `record`.
Only a successful fast-test phase saves its database, even if a later gate fails.
Cache keys isolate architecture, precision, dependencies, configuration and test
assets. Library, test and package-script changes also invalidate the database:
source-inspection tests and child interpreters have dependencies Testmon cannot
trace. Repeated runs and documentation edits can reuse passing results; reference
and release lanes still execute in full. STATADIC tests cache compiled JAX programs in
`$XDG_CACHE_HOME/statadic/jax-tests` (default `~/.cache`); delete it to reclaim
space. Optional reference and qualification suites remain explicit package
commands. `just python::build` writes a wheel and sdist to ignored
`src/python/dist/`.

The [documentation app](apps/docs/README.md) builds the learning site and native labs.

The internal [studies app](apps/studies/README.md) owns reproducible statistical
campaigns; [performance](apps/performance/README.md) owns isolated timing and
memory measurements. Both use the standard-library evidence package. None is a
runtime dependency of the STATADIC wheel. Statistical qualification and benchmark
results retain separate acceptance policies.

See [STATADIC's README](packages/statadic/README.md) for public workflows and package-only
development, and [ARCHITECTURE.md](ARCHITECTURE.md) for workspace ownership.

`just release-check` adds retained-artifact, installed-wheel reference,
float32/float64 reference and dedicated float64 qualification lanes outside the
checkout. It records their distinct scopes in `build/release/manifest.json`.
It is deliberately separate from fast pull-request checks. Hosted releases run
`just release-preflight`, which omits long statistical qualification and archives
its evidence before uploading. A preflight archive is not a scientifically
qualified release. Run full qualification on a persistent local machine and
retain its complete evidence before making that claim.

## Repository scripts

AgentSync's collision preflight is a zero-dependency PEP 723 script under
scripts/agent_preflight/. Its interpreter comes from Nix; it does not require the
Python application workspace. Run it through the root agents commands.

## Workflow scripts

Planning and reference commands use scripts/workflow/script.py with its own
PEP 723 metadata and sidecar lock. It runs independently of the application
workspace; use the root creation, sync and check commands.
