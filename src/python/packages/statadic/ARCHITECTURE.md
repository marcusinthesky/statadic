# STATADIC module architecture

## Scientific role and family

The public tree uses scientific responsibility first and a cohesive mathematical
or statistical family second. Each operation has one implementation owner;
documentation connects overlapping workflows. A family becomes a package when
it contains independently useful components, not when it crosses a line count.

| Area | Owns |
| --- | --- |
| `core` | Shared semantic, equation, matrix and result contracts |
| `optimize` | Numerical algorithms, solver adapters and numerical evidence |
| `sample` | Empirical sample representation and preparation |
| `geometry` | Ground geometry and derived embeddings, medians and barycentres |
| `discrepancy` | Comparisons between measures; energy specializes MMD |
| `association` | Pure association, decomposition and group-comparison statistics |
| `operators` | Reusable structured numerical operations |
| `model` | Statistical specifications and family equations |
| `estimation` | Point estimation and retained fitted state |
| `inference` | Sampling uncertainty, hypothesis tests, diagnostics and resampling |
| `testing` | Explicitly loaded reusable contract batteries and fixtures |
| `api` | Host workflows composing aligned inputs, existing scientific owners, reports and prediction artifacts |

A model describes a statistical family; an estimator computes a fitted state;
a numerical algorithm solves its mathematical problem; inference attaches a
sampling interpretation. IV identification, 2SLS estimation, numerical QR and
an overidentification test therefore have different owners.

## Dependency direction

Dependency order is an explicit module DAG, not a numeric rank assigned to each
root directory. In particular, discrepancy consumes `geometry.ground`, while
`geometry.barycentre` consumes discrepancy. Both relationships are necessary.
The workspace's `tach.toml` records this DAG, including type-only and string-import
edges. `just python::architecture` checks it alongside cross-package import
directions and public interfaces. Tach checks within distributions; the repository
guard covers application imports into STATADIC and rejects hidden dynamic imports and
import-system mutation in shipped STATADIC source. These static checks do not resolve
arbitrary runtime reflection. Tests may inspect private seams but STATADIC tests cannot
depend on internal applications. See the
[architecture tooling](../../scripts/architecture/README.md) for policy ownership,
scope and maintenance.

Every public module declares a literal `__all__`: only its listed names are
public, unlisted top-level definitions start with an underscore, public signatures
and exceptions use public types, and cross-module private helpers require
visibility-scoped Tach interfaces.

Root and compound namespace initializers contain teaching maps, not broad eager
reexports. A family initializer curates its own public surface. This avoids
loading barycentres while importing ground geometry and avoids module/function
name collisions. Use ordinary explicit imports; no lazy importer is needed.

Shared component operations have supported interfaces at their lowest rightful
owner. Private helpers remain inside their feature. Certified core constructors
are producer interfaces, not a reason to publish arbitrary implementation details.
Shared parameter inference consumes structural capabilities rather than an
exhaustive union of concrete fit families.

### Composable public workflows

`api` is an additive orchestration layer above the scientific owners. Its five
workflow entry points are Regression, Ordination, Estimand, Hypothesis and Report.
It does
not move fitting into `model`, replace native fit carriers, or redirect pure
numerical consumers through host preparation. Its family modules expose the
user lifecycle; its initializer remains a navigation map. Scientific
implementations must never import `api`; Tach enforces this dependency direction.

`api.regression.Regression` is the single outcome-model constructor. It owns
predictor terms, response, law, structure and estimation choices; pure response
laws do not own another predictor grammar. Its family package dispatches into
the existing linear, GLM, IV, panel, dyadic, survival and geometric owners.
Shared covariance requests and result conveniences belong to
`api.inference`; relative precision is an input role in `api.data`. Spatial
and panel workflows own their method-specific requests and results. This keeps
shared contracts independent of any particular regression family.

Named `Regression` constructors fix a small, typed composition and return that
same canonical model. They carry no separate runtime identity or inference
defaults. The internal studies catalogue owns declared reference mappings for
exact scientific variants and residual numerical operations; actual execution
belongs to the existing reference runner. Documentation projects these mappings
and their source/proof/study links. The installed library never imports the
catalogue, and a numerical method's mapping grants no inherited variant coverage.

Host `Hypothesis` values describe propositions; `core.statistics.Null` remains
the independent type-level calibration axis. Named restrictions resolve against
actual fitted parameter names, while contrast-derived restrictions also retain
design identity. Parameter tests and diagnostic recipes share `TestOutcome`
reporting without replacing native numerical carriers or influence masks.
BG, DW and HAC require an explicit ordering choice. Ordering gathers aligned
numerical contributions without refitting or replacing the retained fit;
reported lag units are observation positions, not calendar intervals.

`api.roles` owns named input roles and conflict-aware composition. Designs and
model specifications declare their `input_columns` directly; adapters never
discover scientific roles through attribute-name conventions. `Observations`
normalizes raw numeric columns once and exposes immutable array state.
`api.prediction` owns host prediction preparation shared by live workflows and
artifact persistence. Plotting consumes prediction callables; binding a scale is
ordinary callable composition rather than a second predictor hierarchy.

`api.data` owns graph-independent panel layout and node/outcome input carriers.
The `api.spatial` compound namespace is a navigation map; result owners are its
public `cross_section`, `panel`, `observed` and `graph_regression` modules.
This lets ordinary panel prediction consume named graph inputs without loading
every spatial result or creating an import cycle. Live `PanelData` prediction
evaluates WX on complete node covariates before gathering outcome rows. It rejects
conflicting duplicated predictor columns; an omitted outcome-side copy is allowed.

`Model[InputT, PreparedT, ResultT]` shares preparation and fitting without
assuming every input is a regression table. `Ordination` constructs a
representation and implements the same internal term-preparation hook as scalar
predictors. Fitted coordinate maps live in `api.ordination_terms`, below live
models and artifact codecs, so direct composition and frozen predictors share
one implementation. The linear-Gram path keeps feature SVD; kernel, distance and
shortest-path paths retain their own centered-Gram and query state.
`api.estimand.Estimand` composes a closed `Covariance` or `MomentConditions`
quantity with an explicit numerical estimator, without inventing a response
model. Covariance bricks live in `estimation.covariance`; moment results retain
their GMM/GEL owners. Preparation belongs to one model instance.
`Result[FitT, ContextT]` retains actual numerical evidence and host identity;
prediction and inference belong to the scientific family that supports them.
`api.fit_statistics` owns `LikelihoodSummary`, `FitStatistics` and the information
and least-squares R-squared metrics; a family supplies a summary through
`likelihood_summary()` and `Result.fit_statistics()` derives the metrics once.
`api.capabilities` states `SummarizesLikelihood` and `ReportsFitStatistics`.
`core.statistics.PopulationEstimand` is the population-estimand root, separate
from the `api.estimand.Estimand` workflow; `api.design.LinearPredictorContrast`
is the estimand of a predictor contrast.

Compatible estimator bricks implement the narrow `Estimator[ProblemT, FitT]`
capability. The brick owns static estimator options, the numeric problem owns
data arrays, and the native fit owns numerical diagnostics. Bricks live beside
their single scientific implementation, not in a parallel facade hierarchy.
This separation permits JAX transformations over qualified numerical problems
while keeping labels, learned host preparation and rendering outside tracing.
Pytree registration alone does not qualify a new transformation or derivative.

The common input is one aligned outer observation axis, whose columns may hold
scalars or empirical probability measures. Measure support/count storage and
atom masses belong to `sample`; labels and named roles stay in host context.
Atom masses, weights between whole measures, graph weights and regression
precisions retain separate meanings. Pairwise geometry keeps its existing
matrix/estimate carriers and concrete construction evidence. Runtime workflows
cannot recover erased mathematical laws by inspecting generic annotations.

Prepared representations are reusable: the same measure geometry supports
qualified ordination/group comparison and frozen landmark features for scalar
regression. Prediction uses training preparation. A precomputed square matrix
alone does not define cross-distances or a prediction procedure. Shared
capabilities do not imply universal likelihoods, prediction signatures or
inferential validity. New methods require their own mathematical contracts and
qualification before acquiring public capabilities.

Geometric Regression has an explicit full design and reduced-column hypothesis.
`association.geometric` owns projection sums of squares; `inference.geometric`
owns supplied response and Freedman–Lane schedules. No automatic ANOVA type or
learned-outcome-geometry calibration is implied. Spectral positive-subspace
approximation is explicit and retains the signed spectrum. Frozen Isomap queries
attach to the training graph; disconnected training graphs fail preparation.

## Covariance has three distinct targets

`operators.covariance` constructs a matrix from supplied distances and marginal
variances. `estimation.covariance` estimates the covariance of random observations,
including shrinkage and tuning. Inference covariance estimates uncertainty of
retained fitted parameters. PSD feasibility repairs belong to `optimize.psd`.
Shared long-run arithmetic remains below fitting because GMM needs it too.

## Numerical and compatibility boundaries

Precision belongs to the caller. Numerical inputs retain their floating dtypes;
ordinary JAX promotion governs mixed inputs. Generated numerical constants use
the operation's inputs, while structural masks and indices retain their own
boolean/integer meanings. The library neither changes global precision settings
nor chooses a floating payload dtype. Untyped inputs follow JAX's native defaults.

Host preparation checks identities, schemas and static options once. Numerical
kernels receive immutable array state and return fixed-shape numerical evidence.
Pyrefly checks structural interfaces and semantic brands. The required isolated
shape lane checks native matrix, transform, sampler and projection contracts,
including rejected compositions. Native dimensions carry shapes through fields
and factories; jaxtyping strings express runtime dtype and shape constraints.
The opt-in jaxtyping/beartype import hook and Hypothesis tests complement static
checks without claiming a proof of numerical validity. Full-package experimental
stub coverage remains an audit rather than a guarantee.
Do not add tracer-dependent Python assertions or validation to pytree rebuilding.
Precision alone does not veto computation. Numerical failures and tested-domain
limitations remain distinct from convergence and statistical assumptions.

Reusable objectives and solver loops have stable callable identities and receive
changing arrays as explicit arguments. Only genuine algorithm choices are static.
Measure compilation reuse on repeated equal-shape/equal-dtype calls; a nested
function inside one stable traced kernel is not itself evidence of retracing.

The family organization changes canonical import paths, not estimands, samples,
normalization, solver objectives, tolerances, precision ownership or random-key
schedules. Row-index resampling and dyadic node multiplicities remain distinct
contracts. Fit convergence, identification and inferential validity remain
separate evidence. Preview status is documented at the real owning feature;
empty future taxonomy directories are not shipped.

The [module guide](docs/module-organization.md) records canonical imports and the
intentional namespace break, including historical Python qualified names.

Dense least-squares adapters compose JAX's reduced QR and triangular solve,
with separate rank and first-order-residual diagnostics. Explicit Q keeps the
triangular factor independent of the raw Householder buffer: the pinned CPU
vendor's ORMQR temporarily mutates that nominally read-only input, and the
Lineax QR path exposed intermittent corrupted batched refits. This choice
preserves full-rank least squares and wide-system minimum norm at the cost of
materializing Q. Its compilation, memory and execution costs require their own
measurements; other Lineax and Optimistix adapters retain their existing owners.

Affine GMM uses a scaled, column-equilibrated weighted least-squares solve rather
than a normal-matrix pseudoinverse. Its returned weighted-design rank and moment
map belong to that same factorization; fit-bound covariance uses the retained map
and validates the original long-run variability. The supplied weight remains
available beside its effective symmetric working weight. A failed intermediate
weight or coefficient solve makes the lane unavailable, rather than silently
returning a preceding scientific fit. These numerical availability and rank
choices are explicit corrections, not consequences of the family layout.

## Development direction

Future work includes discriminant and metric learning over Wasserstein geometry,
PCoA and Wasserstein DISCO/ANOVA, learned linear embedding transforms targeting
barycentric projection or spatial likelihood objectives, spatial survival and
panel models, and broader `dcor`/POT parity. These are design requirements for
incremental development, rather than claims of implemented support or a reason
to create empty modules. The current [capability guide](docs/capabilities.md)
remains authoritative for qualified operations.

### Compose learning from scientific components

Expose useful numerical objectives and retained state independently of eager
validation, optimization and report assembly. A learned linear map should compose
with existing ground geometry, discrepancy, barycentre and likelihood operations
without copying their algorithms or depending on pipeline configuration. Reuse
the existing sample/projection contracts before introducing another carrier.
Fitted transform state belongs with estimation, geometric application with its
geometry or operator owner, numerical algorithms with optimization, and sampling
interpretation with inference. Choose a concrete family when implementing a
retained use case; this direction does not introduce a generic learning framework.

Learning objectives must state scale, normalization, rank and other constraints
that prevent trivial optima. A rank-reducing map can change separation and other
geometry laws. PCoA, kernel and DISCO consumers must request their actual law
requirements: a Wasserstein name alone does not establish squared-distance CND,
a PSD Gram matrix or a valid energy-test reference. Any spectral correction,
regularization or replacement objective is an explicit scientific choice.

### Qualify each transformation and derivative target

Keep numerical inputs and outputs on the JAX graph where learning needs them;
host conversions and dynamically selected output dimensions belong at documented
preparation or presentation boundaries. State JIT, batching, forward/reverse
derivatives and higher-order derivatives separately. Distinguish differentiation
of an objective, a finite optimization iterate, an optimal value and a converged
solution. A compiled solver establishes none of the latter derivative contracts
by itself.

Assignment ties, active-set changes, repeated eigenvalues, rank changes and
boundary solutions need explicit treatment. Validate gradients against analytic
or independent numerical witnesses, including coincident points and relevant
boundary cases. Exact and regularized transport remain separate methods;
regularization must not silently replace an objective to obtain smoothness.

### Make unequal empirical supports a native design target

Shipped source imports JAX numerical APIs only: NumPy and SciPy are independent
test references, never runtime or type-only imports. Tach's explicit external
bans and the source AST guard run in `just python::architecture`; dynamic
loading, execution and FFI/process escape patterns are also rejected. These
checks are architectural review tools, not a sandbox for hostile Python.
Custom derivative rules require a named exception in that guard. Smooth
objectives use native JAX differentiation; the two PSD spectral JVPs and the
first-order converged GEL envelope rule have explicit mathematical limits.

Measure-level methods should support different empirical support counts through
the shared sample representation and explicit masses where needed. Preserve true
counts independently of storage capacity. `sample.packed.PackedClouds` is the
single empirical-distribution carrier, including atom probabilities. Its dynamic
pytree leaves are data `(*cloud_axes, capacity, D)`, integer counts
`(*cloud_axes)` and masses `(*cloud_axes, capacity)`. Numerical construction
checks structure; checked factories establish probability-domain values.
Differentiate floating coordinate, transform or mass-logit parameter pytrees
while keeping count leaves fixed. Packing alone does not extend an algorithm's domain.
In particular, equal-cardinality uniform assignment remains a specialized path,
not a general solver for unequal supports. Unequal support counts and unequal
total transported mass are distinct contracts.
Atom masses, barycentre source weights and regression precision weights also
retain their separate meanings.

Do not equalize counts by silent truncation, resampling or adding probability
mass at padded atoms. Choose masked, bucketed, chunked or other device-native
execution appropriate to the operation, avoiding mandatory global padding to the
largest cloud when it wastes work. Document static capacities, preparation costs,
memory and compilation behavior; benchmark realistic asymmetric sizes as well
as equal-size inputs. Test that inactive storage cannot affect values or
qualified gradients. Paired observation axes still require alignment even when
each observation is itself an empirical measure with a different support count.

### Preserve statistical design as geometry becomes learned

Record what data fits a projection, kernel or spatial graph, what is frozen for
evaluation, and what is refitted under a declared resampling design. Learned
geometry does not inherit fixed-geometry test validity automatically. In
particular, learning W from outcomes changes the interpretation of a likelihood
conditioned on an exogenous W and requires its own identification and inference
argument.

Spatial survival and observed-panel methods reuse aligned unit/time identities,
time-varying W operators and existing fitting/inference boundaries. Preserve
method-specific event, censoring, truncation, risk-set and observation-time
semantics in their actual model and sample contracts. Unequal cloud supports,
unbalanced panels and absent risk-set members must not be conflated or encoded
as observed zeros.

`estimation.spatial` owns Gaussian observed-subvector panel likelihoods;
`model.survival` owns AFT equations and dependence specifications;
`estimation.survival` owns independent and joint Gaussian log-time fitted states.
The shared Gaussian-tail numerical algorithm lives in `optimize.integration`.
Host `api.survival` prepares censoring patterns and graph/effect alignment;
`inference.survival` consumes the actual free-parameter information. Independent
spell likelihoods and dependent spatial likelihoods do not share a fabricated
rowwise-score contract. Fixed-draw kernels remain transformable; adaptive
integration refinement is an explicit eager controller retaining every attempt.
Positive spatial delayed entry, frailty and start–stop histories remain separate
scientific extensions. See [survival workflows](docs/survival-models.md).

### Establish parity and performance per method

Expand `dcor`/POT comparisons under explicit estimand, bias correction, support
mass, rooting, normalization and solver conventions. Reference dependencies stay
optional and outside the runtime closure. Agreement with an external package,
statistical validity, gradient correctness and performance are separate claims.
Measure preparation/transfers, compilation, synchronized execution and memory
for retained workloads. Preserve efficient sufficient-statistic reductions when
promoting application methods into STATADIC; JAX syntax alone is not a speed claim.
