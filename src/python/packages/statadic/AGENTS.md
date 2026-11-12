# statadic: JAX Distance Correlation and Energy Statistics

JAX implementations of distance correlation, energy statistics, model fitting
and resampling with explicit transformation and precision contracts.

Model workflows use `model.estimate(data) -> result`, then family-specific
`result.diagnose`, `result.predict`, `result.infer` and `result.report`.
`estimate` is the only estimation method; do not add `fit` aliases. Reusable specifications
are frozen. Numerical estimator bricks live beside their existing scientific
implementations and use typed problems; host preparation and reporting remain
outside JAX transformations. See the README quick start and [composition reference](docs/reference/composition.md).
Use `api.inference` for shared covariance requests, `api.data.RelativePrecision`
for that input role, and each workflow family for its own model-specific types.

## Installation and development

Use the [package development workflow](../../../../CONTRIBUTING.md#package-development).
Python 3.13, an installed package (editable or wheel), and the declared dev group
are required. Run `just lint`, `just typecheck`, and the package test recipes;
`just test-artifacts` checks both wheel build routes and standalone sdist
development outside the checkout. Keep pytest fixture loading explicit and
statistical reference dependencies optional.

## Module ownership

Organize by scientific role, then cohesive family, and import from the owning public
family rather than root aliases. The [architecture](ARCHITECTURE.md) defines roots and
the dependency DAG, and [CONTRIBUTING.md](../../../../CONTRIBUTING.md#module-ownership)
has the guards and conventions. The [module migration guide](docs/reference/upgrading.md)
records canonical paths.

## The contract layer

Three shared contracts carry what a value *means*, independently of the stage
that produced it. Each is the reason its waist holds:

| contract | carries | producers |
|---|---|---|
| `core.matrices.DMat` / `SpaceDMat` | a dissimilarity's space, law, and warrant | `geometry.ground`, `discrepancy` |
| `core.statistics.RealizedEstimate` / `CalibratedTest` | what a *number* targets, by which scheme, against which null | `association`, `inference` |
| `core.equations` — ψ, `J`, `Ω` | what an *estimator* solves, and the meat's long-run scheme | `operators`, `estimation` |

**Shared contracts are independent of their producers.** The contract layer can name
the inputs and outputs of a composition without importing statistical
implementations. Typed matrix refinements and generic sandwich composition
remain here; model-specific fitting and uncertainty producers do not.

### Descriptors carry axes; carriers carry descriptors

A **carrier** is a pytree holding arrays (`DMat`, `CalibratedTest`,
`VariabilityMatrix`). A **descriptor** is a zero-field marker holding only type
parameters (`MatrixContract`, `EstimateProvenance`, `TransformContract`,
`MathematicalSpace`).

> A carrier takes **at most three** type parameters — one value type plus two
> descriptors. A new semantic axis goes *inside a descriptor*, named once with a
> `type` alias, never onto the carrier.

Focused typing and contract tests exercise it; the completed migration no longer
keeps an exact AST census. The
cost this controls is invisible one axis at a time: without the budget, every
axis multiplies every carrier signature and every annotation naming one. Before
the descriptor migration `CalibratedTest` took 7 parameters and one field
annotation ran to seven lines in two modules; it now takes 3 and one line.

[`docs/index.md`](docs/index.md) indexes the same code by the **inferential
question** (independence, homogeneity, equivalence, estimation, calibration).

> **Caveat.** `energy_distance` requires `metric`. `metric="angular"` normalises
> before comparing and cannot see scale: `energy_distance(X, 3X) == 0` under it.
> Pass `metric="euclidean"` wherever identification matters. The
> [method catalogue](docs/reference/method-catalogue.md) has the full table.

Equivalence and mixture entry points (`equivalence_test`, `corrected_mixture_test`,
`studentized_mixture_statistic`) are listed in the [method catalogue](docs/reference/method-catalogue.md).

## Performance

Caller arrays and ordinary JAX promotion determine floating precision. Do not
pin floating payload casts, change process precision, or reject a computation
solely for its dtype. Validate host identities/options before transformations;
keep numerical failures and qualification diagnostics as array evidence. Prefer
module-level numerical functions with explicit data arguments over per-call
transformed closures. Hoist only real reuse boundaries, not every local function.

Use the [checked capability and measurement scope](docs/reference/transformation-support.md).
The original `statadic_qualification` DVC stage is not present in this checkout.
Use the internal performance application's repository Just commands for retained
CPU measurements, and the studies application for statistical campaigns. Keep
compilation, synchronized execution and compiler allocation estimates distinct.
Do not infer GPU speed or fitted-solution derivatives from JIT alone.

## Documentation

- [AGENTS.md](AGENTS.md) — Agent conventions
- [notebooks/](notebooks/) — Monte Carlo simulations
- Full API docs in `docs/`

## References

- Székely & Rizzo (2013). *Energy statistics: A class of statistics based on distances.*
- Székely & Rizzo (2004). *Testing for equal distributions in high dimension.*
