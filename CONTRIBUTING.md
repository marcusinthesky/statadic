# Contributing

## Setup

Use Linux x86-64 or ARM64 with Git and Nix. Keep uv available for Copier updates.
Initial setup downloads locked dependencies; later checks use prepared inputs.

```bash
git init -b main
bash infra/nix/install-devenv.sh
devenv shell
just setup
```

Run `just --list` to discover this project's commands.

## Your first plan

```bash
just plan-new project "Project plan"
just task-new .context/plan/project first-step "Make the first useful change"
just sync
just check
```

Open [ROADMAP.md](ROADMAP.md) and follow the plan link. Edit the task's frontmatter
and synchronize to refresh its parent's table. Each parent shows immediate
children; its own status is authored separately.

## Your first reference

```bash
just reference-new first-source "First source"
just sync
just check
```

Edit the source's flat CSL metadata in
`.context/reference/first-source/README.md`. Its directory is the citation key.
Synchronization derives `src/latex/refs.bib` from these records.

## Try the code

```bash
just python::run
just lean::build
just lean::blueprint-web
just latex::build
```

## Package development

Work on the `statadic` package in a dedicated environment, so that it does not prune the
shared workspace environment. From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$(mktemp -d)/venv"
uv sync --project src/python --package statadic
cd src/python/packages/statadic
just lint
just typecheck
just test-fresh tests/core/test_domains.py
just test-artifacts
```

These commands install the declared development dependencies, including Ruff and
Pyrefly, and no unrelated applications. Just and uv come from the repository
development shell. `pyproject.toml` is the version authority, and
`statadic.__version__` reads the installed distribution metadata, including editable
installations. Importing an uninstalled source directory is not a supported workflow.
Update `[project].version` when preparing a release, as the
[release guide](infra/release/README.md) describes. Contributor linting and typing use
Ruff 0.16, Pyrefly 1.4 and JAX 0.11. The runtime JAX series stays bounded pending
scientific-stream qualification.

A standalone package directory and an extracted sdist also work on Python 3.13 with
uv and Just installed. Run `uv sync` in that directory, then the same `just`
commands. The sdist includes the tests, scripts and configuration this needs.

### Typing

`just typecheck` runs two checks: strict structural checking of source, tests and
scripts, and `just typecheck-shapes` in an isolated environment with the pinned shape
stubs. The second checks native shape contracts in matrices, transforms, samplers and
projections, then runs positive and negative shape fixtures. `just
typecheck-shapes-all` audits the full package against the experimental stubs, and
`just typecheck-shape-coverage` separates return shapes Pyrefly infers from those it
only compares with a local annotation and those it does not check. Failures propagate
to the command's exit status.

Use native dimension parameters when shapes must survive factories, fields or mapping.
Import the runtime-safe vocabulary from `statadic.core.typing`. For example,
`Float[ShapedArray[["Rows", "Width"]], "n d"]` combines a static shape with a runtime
floating-dtype and shape check. Quote native dimension references, because jaxtyping
evaluates annotations without function-local type parameters. For ordinary local array
relationships, direct jaxtyping imports and `@static_jaxtyping("n d")` stay readable.
Shape strings are runtime contracts, not comments. Runtime checks use jaxtyping and
beartype through the opt-in import hook, and static checks do not prove numerical
validity.

Matrix annotations group law and construction origin in
`DMat[Domain, MatrixProvenance[Law, Origin]]`. An optional third parameter keeps the
array shape and constructors normally infer it. This holds semantic provenance and
shape information within Pyrefly's generic budget.

### Tests

Run `just test` for cached fast tests or `just test-all` for the full suite. `just
test-layout` checks module and package ownership without importing JAX, and pytest
enforces it automatically. The [test ownership
guide](src/python/packages/statadic/tests/README.md) defines the cross-module
categories. The layout enforcer is a reviewed fork pinned to an immutable HTTPS Git
commit. The workspace `just python::statadic::test-fresh` runs the complete fast
suite without Testmon. Fixtures load explicitly from `statadic.testing.fixtures`
after the runtime annotation hook is installed, so installing STATADIC alone does not
register a pytest plugin.

`just test-artifacts` builds a wheel and sdist from a temporary package snapshot,
rebuilds a wheel from that sdist and checks both installed runtime closures in
separate temporary environments outside the checkout. It also runs lint, types and
the typing tests from the extracted sdist, and may download build, runtime and
development dependencies. It leaves out the optional reference group and does not
publish. License selection and publication policy remain maintainer decisions.

Optional diagnostics, from the repository root:

```bash
just python::statadic::test-coverage
just python::statadic::test-monitor
```

Coverage reports missing lines and branches in the complete fast suite and writes an
HTML report to `src/python/packages/statadic/htmlcov/index.html`. There is no minimum
percentage. Use the report to find meaningful gaps, because numerical and statistical
correctness need independent evidence. Both commands accept test paths and pytest
selectors, and a restricted run reports only that selection. Monitoring installs the
optional `profiling` group and records test resources in the ignored `.pymon` SQLite
database. Run it serially (the recipe uses `-n 0`) to avoid concurrent writers.
Ordinary test commands disable the monitor plugin. Use the performance app for
controlled JAX compilation and synchronized timings.

Independent mathematical witnesses run in the ordinary test tiers. Live comparators
stay outside the runtime wheel and default environment. Prepare their locked
environment and run the required lane from the repository root:

```bash
just setup-references
just validate-references
```

`reference` selects comparisons against independent software, analytic results,
published values and trusted fixtures. `external_reference` selects live comparisons
against optional software, and those tests also carry `reference` and
`requires_optional("<distribution>")` so they skip when the dependency is absent. The
required lane rejects missing comparators, empty selections, skips and expected
failures. Package-only contributors can install the reference dependency group and use
`just test-external`. The [release guide](infra/release/README.md#validation-lanes-and-ci)
describes the precision, qualification and CI lanes.

### Compatibility checks

The contributor gate runs Python 3.13 at the locked resolution. Endpoint checks run
46 targeted tests per environment on Python 3.13 and 3.14, at lowest-direct and highest
dependencies. They do not qualify the full fast suite on 3.14 or every dependency
combination. For a local installed-wheel check, run:

```bash
devenv shell -- just python::statadic::compat -e 3.13-lowest -- tests/api/test_inputs.py
```

The locked contributor `compat` group supplies tox and tox-uv in an isolated launcher
environment, so its uv does not shadow the pinned repository tool. Test environments
resolve lowest-direct or highest dependencies independently of the contributor lock,
and preparing them may download interpreters and wheels. The optional
`x32-3.13-highest` and `x32-3.14-highest` CPU environments are also available. Run
named files in each environment one after another. This lane stays outside `just
check`, and a passing subset establishes only that subset's coverage.

Random streams keep STATADIC's keys, shapes, precision and tested statistical
baselines, but floating draws can differ bit for bit across JAX patch releases.
Reproducing a study needs its exact dependency versions, precision, source and platform,
so keep the frozen environment with the study artifacts. Key-taking APIs accept scalar
typed JAX keys and legacy `uint32[2]` keys. Typed Threefry keys are tested under
JIT and vmap, and `cell_key` keeps its label-derived schedule. Other PRNG
implementations follow JAX's semantics, including its [documented batching
limitations](https://docs.jax.dev/en/latest/jax.random.html#advanced), and an equal
seed does not promise the same stream across implementations.

### Module ownership

The package [ARCHITECTURE.md](src/python/packages/statadic/ARCHITECTURE.md) defines the
root responsibilities and the nested dependency DAG. Run `just python::architecture`
from the repository root for Tach's internal and external checks, the source AST guard
and its rejection fixtures. Each governed runtime module forbids NumPy and SciPy,
including type-only imports, while tests may keep independent references. Dynamic
loading, execution and FFI or process escape patterns are rejected. The AST guard is a
review tool, not a hostile-code sandbox. The [module migration
guide](src/python/packages/statadic/docs/reference/upgrading.md) records canonical
paths and clean import breaks.

Numerical adapters in `statadic.optimize` have result-bearing variants:
`least_squares_result`, `minimise_bfgs_result`, `pgd_simplex_affine_result` and
`simplex_qp_result`, which solves convex quadratic objectives on a possibly capped
probability simplex and keeps feasibility, KKT and optimality-gap evidence. Inspect
status and residual evidence before using an iterate. The value-returning functions
remain, and affine PGD supports JIT and batching but not reverse-mode differentiation.
`RBFKernel` and `sample.reduction.Projection` are immutable pytrees that keep learned
numerical state dynamic, while metric and kernel laws stay static declarations. The
RBF kernel uses squared Euclidean differences directly, which keeps derivatives finite
at coincident points.

`discrepancy.transport.wasserstein_1d` and `wasserstein_1d_power` support weighted
`PackedClouds` with unequal support counts and broadcast cloud axes. The latter returns
the unrooted objective, and sliced transport uses masses from the same carrier under
one shared projection design. Exact uniform multivariate assignment keeps its
equal-cardinality contract. `statadic.core.equations.sandwich_covariance` returns the
limiting covariance for a mean equation, and `parameter_covariance` scales it by the
observation count with a separate finite-sample correction. Both accept derivative
panels or aggregate mean Jacobians. The [mathematical
contracts](src/python/packages/statadic/docs/concepts/geometry.md)
state normalization and validity limits.

## Find your way

- [Project context](.context/README.md) explains the installed workflows.
- [The Python workspace](src/python/README.md) explains code and commands.
- [The Lean workspace](src/lean/README.md) explains code and commands.
- [The Latex workspace](src/latex/README.md) explains code and commands.
- [AgentSync](.agents/README.md) manages shared instructions and native configuration.
- [Architecture](ARCHITECTURE.md) explains ownership and updates.

Edit source, run `just sync`, then `just check`.
Formatting and synchronization are explicit writes. External links run separately
through `just links`.

For version tags, PyPI publishing and GitHub releases, see the
[release guide](infra/release/README.md).

## Update

Commit local work, then run:

```bash
env -u PYTHONPATH -u PYTHONHOME uvx --from copier==9.16.0 copier update --vcs-ref v0.1.0-alpha
```

Review the diff and conflicts, run setup, sync and check. The explicit reference pins
the alpha release. Adding capabilities preserves saved selections. Removing them
requires manual migration because Copier can remove previously owned paths.
Actual records remain project-owned.

This project started with [Exordio](https://marcusinthesky.github.io/exordio/); see the [alpha release](https://github.com/marcusinthesky/exordio/releases/tag/v0.1.0-alpha).
