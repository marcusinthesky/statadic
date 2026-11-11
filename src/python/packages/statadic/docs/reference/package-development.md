# STATADIC: statistical computing with JAX

STATADIC offers ready-to-use statistical models and reusable components. Start with [installation](../start/installation.md), the [common workflow](../start/workflow.md), and a [first regression](../tutorials/first-regression.md). The [OLS chapter](../tutorials/ols/index.md) develops a complete analysis; [how-to guides](../guides/index.md) cover specific tasks. The [documentation index](../index.md) links to concepts, reference contracts and evidence.

JAX implementations of distance correlation, energy statistics, model fitting
and resampling with explicit transformation and precision contracts.

## Installation and supported environment

STATADIC declares Python 3.13–3.14. Install from this checkout using uv:

```bash
uv venv --python 3.13
uv pip install ./src/python/packages/statadic
```

The runtime depends on CPU JAX and numerical libraries, without pipeline apps,
production data or independent statistical reference packages. GPU installation
and numerical precision remain caller choices; the repository defaults to CPU.
STATADIC preserves floating inputs and ordinary JAX promotion, without internal
precision overrides. Caller-selected precision does not establish numerical
accuracy: inspect retained solver and reference-qualification diagnostics.
`pyproject.toml` is the version authority, and `statadic.__version__` reads the
installed distribution metadata, including editable installations. Importing an
uninstalled source directory is not a supported development workflow.

## Compatibility checks

The contributor gate runs Python 3.13 at the locked resolution. Retained endpoint
checks cover 46 targeted tests per environment on Python 3.13 and 3.14, at
lowest-direct and highest dependencies. They do not qualify the full fast suite
on Python 3.14 or every dependency combination.

Random-stream compatibility preserves STATADIC's keys, shapes, precision and tested
statistical baselines. Floating random draws are not promised to match bit for
bit across JAX patch releases. Reproducing a retained study requires its exact
dependency versions, precision, source and platform; preserve the frozen
environment with the study artifacts.

For local installed-wheel compatibility checks, use
`devenv shell -- just python::statadic::compat -e 3.13-lowest -- tests/api/test_inputs.py`.
The locked contributor `compat` group supplies tox and tox-uv in an isolated
launcher environment, so its uv does not shadow the pinned repository tool.
Test environments resolve lowest-direct or highest dependencies independently of
the contributor lock; preparing them may download interpreters and wheels.
Python 3.13 and 3.14
are covered, with optional `x32-3.13-highest` and `x32-3.14-highest` CPU lanes.
Run named files in each environment sequentially. This lane stays outside
`just check`; passing a selected subset establishes only that subset's coverage.
Test fixtures under `statadic.testing.fixtures` require contributor pytest and are
loaded explicitly; plain library installation does not require pytest.

## Quick Start

For named models and distribution-valued observations, use the
[composition reference](composition.md). One retained empirical-measure
geometry can feed ordination, group comparison and frozen landmark predictors.
Model specifications are reusable frozen objects; fitting returns a new result.

```python
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression

training = Observations({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.1, 2.8, 4.2]})
model = Regression(response="y", terms=Intercept() + Linear("x"))
result = model.estimate(training)
prediction = result.predict(Observations({"x": [4.0, 5.0]}))
uncertainty = result.infer(covariance=HC("HC3"))
print(uncertainty.report(reference="normal"))
```

Numerical estimator bricks expose `.estimate(problem)` under the existing JAX
transformation contracts. Distances, operators and testing functions remain
available for direct numerical composition:

```python
from statadic.discrepancy.mmd.energy import UNIT_EXPONENT, energy_distance, energy_geometry
from statadic.geometry.ground.metrics import ANGULAR, EUCLIDEAN, cdist
from statadic.inference.resampling.permutation import energy_permutation_test
from statadic.sample.packed import PackedClouds
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

dist = cdist(x, y, metric=EUCLIDEAN)
e_dist = energy_distance(
    PackedClouds.from_points(x), PackedClouds.from_points(y), metric=ANGULAR
)
result = energy_permutation_test(
    x,
    y,
    energy_geometry(EUCLIDEAN, UNIT_EXPONENT),
    num_permutations=9999,
    seed=42,
)
print(f"p-value: {result.pvalue:.4f}")
```

## Package-only development

Inside the checkout, use a dedicated environment to avoid pruning the shared
workspace environment. From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$(mktemp -d)/venv"
uv sync --project src/python --package statadic
cd src/python/packages/statadic
just lint
just typecheck
just test-fresh tests/core/test_domains.py
just test-artifacts
```

These commands install STATADIC's declared development dependencies, including Ruff
and Pyrefly; they do not install unrelated applications. Just and uv are supplied
by the repository development shell. Update `[project].version` when preparing a release; runtime identity needs
no second version literal.
Contributor linting and typing use Ruff 0.16, Pyrefly 1.4 and JAX 0.11.
The runtime JAX series remains bounded pending scientific-stream qualification.

`just typecheck` requires two checks: strict structural checking of source,
tests and scripts, and `just typecheck-shapes` in an isolated environment with
the pinned shape stubs. The latter checks native contracts in matrices,
transforms, samplers and projections, then runs positive and negative shape
fixtures. These four modules are checked in the shape environment rather than
against the ordinary JAX annotations. `just typecheck-shapes-all` additionally
audits the full package against the experimental stubs. `just typecheck-shape-coverage`
separates return shapes Pyrefly infers from those it only compares with a local
annotation and those it does not check, since a gradual value satisfies any
annotation. Stub gaps and call sites
outside the required shape lane remain unresolved; failures propagate to this
investigative command's exit status.

Use native dimension parameters when shapes must survive factories, fields or
mapping. Import their runtime-safe vocabulary from `statadic.core.typing`. For
example, `Float[ShapedArray[["Rows", "Width"]], "n d"]` combines a static shape
with a runtime floating-dtype/shape check. Quote native dimension references:
jaxtyping evaluates annotations without function-local type parameters. For
ordinary local array relationships, direct jaxtyping imports and
`@static_jaxtyping("n d")` remain readable. Shape strings are real runtime
contracts, not comments. Runtime checks use jaxtyping/beartype through the
existing opt-in import hook; static checks do not prove numerical validity.

Matrix annotations group law and construction origin in
`DMat[Domain, MatrixProvenance[Law, Origin]]`. Its optional third parameter
retains the array shape and is normally inferred by constructors. This keeps
semantic provenance and shape information within Pyrefly's generic budget.

A standalone package directory and an extracted sdist are also supported on
Python 3.13 with uv and Just installed: run `uv sync` in that directory, then
the same `just` commands. Ruff policy is self-contained, and the type-checking
recipe explicitly selects the active environment and strict Pyrefly settings.
The sdist includes tests, scripts and configuration needed for this workflow.

`just test-artifacts` builds a wheel and sdist from a temporary package snapshot,
rebuilds a wheel from that sdist, and checks both installed runtime closures in
separate temporary environments outside the checkout. It also checks lint,
types and existing typing tests from the extracted sdist. It may download build,
runtime and development dependencies. It does not install the optional reference
group or certify every scientific method. License selection and publication
policy remain maintainer release decisions; this check does not publish.

## Reference checks

The [validation guide](../validation.md) distinguishes live comparisons,
independent mathematical witnesses and failure-inclusive statistical qualification,
and documents the mandatory reference and scheduled study commands.

Run `just test` for cached fast tests or `just test-all` for the full suite.
`just test-layout` checks focused module/package ownership without importing JAX;
pytest also enforces it automatically. The {download}`test ownership guide <../../tests/README.md>`
defines the explicit cross-module categories. The development dependency uses
the reviewed layout-enforcer fork pinned to an immutable HTTPS Git commit.
The workspace `just python::statadic::test-fresh` runs the complete fast suite
without Testmon. Fixtures are explicitly loaded from `statadic.testing.fixtures`
after installing the package's runtime annotation hook; installing STATADIC alone
does not register a pytest plugin.

Optional diagnostics, from the repository root:

```bash
just python::statadic::test-coverage
just python::statadic::test-monitor
```

Coverage reports missing lines and branches in the complete fast suite and writes
an HTML report to `src/python/packages/statadic/htmlcov/index.html`. There is no
minimum coverage percentage. Use reports to find meaningful gaps; numerical and
statistical correctness still require independent evidence, and tests written
only to increase coverage add no assurance. Both commands accept test paths and
pytest selectors; a restricted run reports only that selection's coverage.

Monitoring installs the optional `profiling` dependency group and records test
resources in the ignored package-local `.pymon` SQLite database. Run it serially
(the recipe uses `-n 0`) to avoid concurrent writers. Ordinary test commands
disable the monitor plugin even when that group is installed. Use the internal
performance app for controlled JAX compilation and synchronized execution timings.

Independent mathematical witnesses run in the ordinary test tiers. Live optional
comparators stay outside the runtime wheel and default development environment.
From the repository root, prepare their locked environment and run the required
lane:

```bash
just setup-references
just validate-references
```

`reference` selects comparisons against independent software, analytic results,
published values, and trusted fixtures. `external_reference` selects live
comparisons against optional software; those tests also carry
`reference` and `requires_optional("<distribution>")`, so they skip cleanly
when the dependency is absent. The required external-reference lane rejects
missing comparators, empty selections, skips, and expected failures. A reference
comparison can have any scope, depending on the boundary exercised by its
assertions. Package-only contributors can install the reference dependency group
and use `just test-external`.

## Module ownership

Organize by scientific role, then cohesive family. Model specifications,
estimators, numerical algorithms and inferential procedures have different
owners. Import from the owning public family rather than root convenience aliases.

The {download}`architecture <../../ARCHITECTURE.md>` defines the root responsibilities and the
explicit nested dependency DAG. In the repository, `just python::architecture`
runs Tach's internal and external checks plus the source AST guard and rejection
fixtures. Each governed runtime module forbids NumPy and SciPy, including
type-only imports; tests may retain independent references. Dynamic loading,
execution and FFI/process escape patterns are also rejected. The AST guard is
a review tool, not a hostile-code sandbox. Numeric ranks per root cannot describe both ground geometry and discrepancy-consuming
barycentres. Root and compound initializers provide navigation; family
initializers curate their own exports. Keep private implementation details
inside their family and expose narrow component operations for real consumers.

The [module migration guide](../module-organization.md) records canonical paths,
clean import breaks, and qualified-name persistence considerations. Do not add
empty future taxonomy branches or duplicate facade trees.

The architecture's development-direction section records design
requirements for learned geometry, unequal empirical supports, spatial
survival/panel methods and broader reference parity. It guides incremental
feature work; current support is documented by each owning family.

Numerical adapters have result-bearing variants in `statadic.optimize`:
`least_squares_result`, `minimise_bfgs_result` and `pgd_simplex_affine_result`.
`simplex_qp_result` solves convex quadratic objectives on a possibly capped
probability simplex, retaining feasibility, KKT and optimality-gap evidence.
Inspect their status and residual evidence before using an iterate. Numerical
convergence does not establish parameter identification or justify uncertainty;
QR reports rank separately and affine simplex PGD reports feasibility separately.
The existing value-returning functions remain available. Affine PGD supports JIT
and batching, with its existing reverse-mode differentiation limitation.

Learned numerical state can remain dynamic: `RBFKernel` and
`sample.reduction.Projection` are immutable pytrees, while structural protocols
describe the capabilities consumers need. RBF uses squared Euclidean differences
directly, preserving finite derivatives at coincident points. Metric and kernel
laws remain explicit static declarations.

`discrepancy.transport.wasserstein_1d` and `wasserstein_1d_power` support weighted
`PackedClouds` distributions with unequal support counts and broadcast cloud axes.
The latter returns the unrooted objective; sliced transport uses masses from
the same carrier under one shared projection design. Exact
uniform multivariate assignment retains its separate equal-cardinality contract.

`statadic.core.equations.sandwich_covariance` returns the limiting covariance for a
mean equation. `parameter_covariance` explicitly scales that quantity by the
retained observation count, with a separate finite-sample correction. These
component APIs accept either derivative panels or aggregate mean Jacobians;
the preview [linear fitting workflow](../linear-models.md) owns the actual
sample, weights and parameter layout for OLS/WLS covariance and inference. See the
[mathematical contracts](../mathematical-contracts.md#estimating-equations)
for normalization, alignment obligations and validity limits.

Key-taking APIs accept scalar typed JAX keys and legacy `uint32[2]` keys.
Typed Threefry keys are tested under JIT/vmap; `cell_key` retains its legacy
label-derived schedule. Other PRNG implementations follow JAX's semantics,
including its [documented batching limitations](https://docs.jax.dev/en/latest/jax.random.html#advanced).
An equal seed does not promise the same stream across implementations.

Below waist 1 anything producing a distance matrix is interchangeable; above it,
nothing knows which produced it. Same at waist 2. That is why novel combinations
— energy divergences standing in for covariance, spatial operators built from
energy geometry — are free rather than boundary violations.

[`docs/index.md`](../index.md) indexes the same code by the **inferential
question** (independence · homogeneity · equivalence · estimation ·
calibration), which is how people actually arrive at the library.

> **Removed in t46**: `statadic.metrics`, `statadic.statistics`, `statadic.wasserstein`,
> `statadic.kernels`, `statadic.permutation`, `statadic.hypothesis`, `statadic.equivalence`,
> `statadic.block_length`, `statadic.mantel`, `statadic.seeds`, `statadic.pit_energy`,
> `statadic._energy_components`. Deleted, not deprecated — see `CHANGELOG.md` for
> the destination of each. No symbol was renamed by the move.
>
> **Caveat worth reading before you copy the snippet above.**
> `energy_distance` requires `metric`. The angular pullback normalises before
> comparing and therefore *cannot see scale*: `energy_distance(X, 3X) == 0` under
> `ANGULAR`. Pass the `EUCLIDEAN` strategy wherever identification matters. The
> declared axioms and `Premetric` brand say so; `docs/index.md` has the full
> table.

## Validity-corrected hedge-quality inference (Paper 2)

The legacy `energy_kernel_permutation_test` / `mixture_approximation_test` read a
**high** permutation p-value as a **good** hedge — an inverted, double-dipping,
non-exchangeable, mis-scaled test. They have been removed; the corrections below
replace them with a composed, defensible statistic spread across the modules
above by topic (not one `corrected` module). **The core public API is
unchanged** (`energy_distance`, `energy_test_statistic`, `energy_permutation_test`,
`kernel_balancing_weights`); these are new entry points alongside it.

| Entry point | Module | Correction |
|---|---|---|
| `inverse_variance_m_eff` | `discrepancy.mmd.energy.balancing` | #1 effective size `1/m_eff = Σ w_k²/m_k` (Kish; not `wᵀm`) |
| `equivalence_test` | `inference.equivalence` | #2 relevance null `H0: E ≥ δ`; reports `Ê` + bootstrap CI + equivalence curve |
| `corrected_mixture_test` | `inference.mixture` | #3 re-optimize `w*` in-permutation OR select-A/test-B split |
| `studentized_mixture_statistic` | `inference.studentized` | #4 studentized (Chung–Romano) mixture statistic |
| `fit_projection` | `sample.reduction` | #5 dim-reduction (fit on split A only) |
| `equivalence_multiplicity_control` | `inference.multiplicity` | Holm (default) or Bonferroni; FWER control requires valid true-null p-values, which is not established for the equivalence bootstrap |
| `optimal_block_length`, `stationary_bootstrap_ci` | `inference.resampling.block_length` | OOS / CI block length: `scheme=stationary` → `D_SB=2g²(0)`; `circular` → `D_MB=(4/3)g²(0)` |

```python
from statadic.inference.equivalence import equivalence_test

# Equivalence-framed, studentized, re-optimized (all three at once):
result = equivalence_test(
    x,
    candidates,
    delta=0.5,
    selection_split=None,
    reoptimize_in_permutation=True,  # full-sample route
)
print(result.energy_point_estimate, result.energy_ci)  # PRIMARY metric
print(result.equivalence, result.equivalence_curve)  # δ verdict + curve
```

The δ-margin is an **explicit, documented parameter** with no consensus standard
for energy distance: the caller must justify it against a domain reference.
`m_eff` is the non-degenerate / studentization normalizer, **not** an
exact-null quantity (the null statistic is degenerate).

## Performance

The [capability and measurement guide](../capabilities.md) states the checked
JIT, batching, derivative and precision boundaries. The CPU fitting and refit measurements were recorded in the source repository.
Its `statadic_qualification` DVC stage and artifacts are not included in this
standalone checkout. The source records separate compilation, synchronized warm
execution, preparation/transfers and compiler allocation estimates. No GPU
speed claim is made from the default CPU environment.

## Documentation

- {download}`AGENTS.md <../../AGENTS.md>` — Agent conventions
- [Interactive labs](../labs/index.md) — executable demonstrations
- [docs/index.md](../index.md) — curated guide by inferential question
- [docs/spatial-models.md](../spatial-models.md) — spatial/panel APIs, reference coverage and limits
- [docs/survival-models.md](../survival-models.md) — censored spells, panel effects and spatial log-time survival
- [docs/gmm-fitting.md](../gmm-fitting.md) — affine moment fitting and actual-weight inference
- [docs/resampling.md](../resampling.md) — paired schedules, retained failures and strict marginal summaries
- [docs/glm-fitting.md](../glm-fitting.md) — binomial trials, Poisson exposure and finite-optimum checks

## References

- Székely & Rizzo (2013). *Energy statistics: A class of statistics based on distances.*
- Székely & Rizzo (2004). *Testing for equal distributions in high dimension.*
