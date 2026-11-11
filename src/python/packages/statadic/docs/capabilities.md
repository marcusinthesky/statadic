# Transformation and performance scope

The table qualifies the implemented preview workflow and its retained numerical
boundaries. A registered result pytree makes arrays transformable; it does not
establish statistical validity, smoothness at a rank change or GPU performance.
All measurements in this work use CPU. Precision belongs to the caller.

The [composable API](reference/composition.md) is an eager host layer over these owners.
Its fitted records preserve numerical state without extending the transformation
claims below. Weighted measure kernels have focused fixed-shape JIT/vmap checks;
packing, named selection, reports and artifact loading stay outside transforms.
Sliced reference/query evaluation requires a matching effective dtype to retain
one projection geometry. New workflow qualification does not follow from these
numerical checks.

| Operation | Eager preparation | JIT and batching | Derivative target | Precision/backend |
| --- | --- | --- | --- | --- |
| [Ordination](reference/ordination.md): feature SVD, kernel/distance Gram and shortest paths | Named preparation, dimension/rank selection and identity checks are eager; fitted maps retain training state | Fixed-shape coordinate kernels, frozen numerical maps and shortest paths have JIT/vmap controls | No general coordinate derivative promise through repeated eigenvalues, rank, axis selection or neighborhood changes | Caller precision; independent NumPy/SciPy and optional sklearn references; dense graph paths use quadratic storage and cubic path work, with no resource-win claim |
| `LeastSquares().estimate(LinearProblem(x, y))` (OLS) | Complete arrays, explicit columns and optional ordered names | Fixed shapes; nested JIT and leading-axis vmap, including invalid lanes | Fitted coefficients with respect to design and outcome, forward and reverse, away from rank/validity changes; independent analytic and finite-difference checks | Float32/float64 CPU |
| `LeastSquares().estimate(LinearProblem(x, y, precision))` (WLS), `LinearFit.predict` | Positive relative-precision weights; exact named-column alignment | Fixed shapes; JIT and leading-axis vmap | No fitted-WLS derivative qualification in this slice | Float32/float64 CPU |
| `IteratedGMM` / `FixedWeightGMM` over `AffineMomentProblem` | Affine moment panel, explicit variability scheme, static bandwidth/iteration budget; supplied fixed weights are dynamic problem leaves | Fixed shapes; JIT and leading-axis vmap | Actual-W objective/first-order diagnostics; no iterated-fit derivative promise | Float32/float64 CPU witnesses; caller precision also governs eager reports |
| `FixedGEL` / `AffineGEL` | Eager fixed or affine moment validation and retained numerical-rank choices | Eager only; problem pytrees do not qualify the host solve for tracing | No fitted-solution derivative promise | Caller precision; reference assumptions explicit |
| `PanelML` / `KKP` over `SpatialPanelProblem` | Complete balanced time-major sample and ordered graph identities | Existing fixed-shape JIT/vmap contracts, with static process, effects and iteration controls | No fitted-solution derivative promise | Float32/float64 CPU; KKP requires constant W |
| Observed spatial panels and survival | Explicit graph nodes, observed outcomes and, for spatial survival, integration designs | Host alignment and refinement are eager; fixed-shape numerical operations have their own contracts | No fitted-solution derivative or statistical-calibration promise | Caller precision; spatial survival integration supports float32/float64; see the [spatial](spatial-models.md) and [survival](survival-models.md) guides |
| `BinomialLogitMLE`/`PoissonMLE` over `BinomialProblem`/`PoissonProblem`, `GLMFit.predict` | Integer response/trials, explicit offset/exposure and ordered columns | Fixed shapes; JIT and leading-axis vmap; finite-optimum and solve outcomes retained per lane | Interior likelihood score/information checked independently; no derivative of the converged fit or separation certificate promised | Float32/float64 CPU |
| Fit-derived covariance, `coefficient_intervals`, `linear_restriction_test` | Consume actual fitted state; reference/correction/df remain explicit | JIT and leading-axis vmap for checked fixed shapes | No blanket derivative promise through rank, status, tail or covariance-domain changes | Float32/float64 CPU witnesses; finite-reference qualification diagnostics in [linear workflow](linear-models.md) |
| OLS residual diagnostics, `ols_influence` | Consume an actual unweighted `LinearFit`; static lags, RESET powers and reference | JIT and leading-axis vmap for fixed shapes | No derivative claim through rank, availability or tail changes | Float32/float64 statsmodels and leave-one-out witnesses; Gaussian-null size study in [residual diagnostics](linear-diagnostics.md) |
| Row schedule factories, `evaluate_replicates`, `refit_replicates` | Fixed row capacity, explicit draw IDs; evaluator closes over frozen preprocessing | Fixed schedule structure; JIT and outer vmap, chunked replicate evaluation | No derivative of discrete schedules or bootstrap calibration promised | Typed/legacy Threefry keys; no cross-PRNG stream identity promise |
| `percentile_summary` | Supplied scalar/vector or limited tuple/dict draws and validity | Fixed structure; JIT and outer vmap | Marginal empirical quantiles; no smooth interval map or simultaneous coverage claim | Caller float32/float64; all requested draws must succeed |
| `PackedClouds` and empirical-distance inputs | Checked point/cloud factories establish masses and counts; numerical constructor checks structure | Three dynamic leaves; JIT/vmap on fixed capacities and broadcast cloud axes | Native AD of floating coordinate/logit parameter trees with counts fixed; linear-MMD analytic gradient and Hessian-action witnesses; no whole-carrier integer-gradient promise | Caller precision; inactive padding is inert |
| `empirical_mean_ground_distances` | Prepared `PackedClouds` | Fixed capacities support JIT/vmap; dynamic ragged preparation stays eager | No blanket smoothness claim at coincident Euclidean points | Caller float32/float64; weighted V-statistic self-pairs |
| Exact uniform transport | Eager validation or explicit fixed-shape JAX door | Existing Optax assignment kernels support JIT/vmap | Discrete assignment does not imply a smooth fitted-solution map | Caller precision; result-bearing cost core reports invalid inputs, nonfinite values and detected power underflow. No warning does not certify assignment accuracy; no GPU speed claim |
| `wasserstein_1d`, `wasserstein_1d_power` | Real locations and nonnegative atom masses, normalized separately; unequal nonempty support counts | Fixed atom capacities; leading batch broadcasting, JIT and vmap; invalid numeric lanes return NaN; zero-mass padding is inert | Powered objective location/mass derivatives away from ordering and quantile ties; rooted distance can be nonsmooth at zero | Float32/float64 CPU; SciPy W1 and independent transport-LP witnesses |
| Weighted sliced transport | Shared projection design and one mass vector per support | Unequal fixed support counts; explicit vmap for separate measures | Inherits one-dimensional piecewise objective boundaries; finite projections retain pseudometric law | Caller precision; no general multivariate unequal-support solver |
| Dynamic `RBFKernel`, `Projection`, energy geometry | Validate structure and declare laws before numerical execution | Registered immutable pytrees keep numerical parameters dynamic through MMD/energy consumers | Projected/kernel objectives checked in forward and reverse mode, including coincident RBF points; no learning or inference procedure supplied | Float32/float64 CPU |
| `simplex_qp_result` | Explicit quadratic/linear objective, scalar weight cap and solver budget; convexity on the simplex tangent space | Fixed shapes; JIT/vmap; independent feasibility, KKT and gap diagnostics | No new derivative of the converged solution promised | Float32/float64 CPU; requested tolerance can remain unmet and must be checked |
| Affine-simplex optimization | Explicit feasible-set and solver configuration | Existing fixed-shape JIT/batching contract | Retained reverse-mode limitation; no derivative-capable solver redesign | Existing solver/status contract |
| PSD repair diagnostics | Square-array shapes; explicit iterative convergence evidence | Array-valued namedtuple diagnostics; semismooth Newton uses a linearized native value/gradient | Two audited spectral JVP rules handle repeated eigenvalues; zero eigenvalues use a selected generalized derivative, not a classical Hessian | Caller precision; explicit convergence evidence remains required |

Static names and options describe schemas and algorithms. Dataset values and
per-lane statuses remain array leaves, rather than compilation-key identities.
Batching means applying vmap to one fixed-shape operation; it does not add a
dynamic ragged dimension to that operation's input contract.

Checked QP results retain interior-point iterates, except for directly certified
uniform or vertex solutions. Numerical optimality does not certify general
exact sparsity: small positive values on inactive faces can change
support-based reference operators at a caller's threshold. Pipeline therefore
retains SLSQP for its anchored W1/W2 projections. Do not infer active-set
membership solely from tiny positive QP weights.

## CPU measurement provenance

The measurements below were collected in the original downstream repository
using its `statadic_qualification` DVC stage and synthetic fixture. That stage and its
`data/shared/statadic_qualification/` artifacts are not part of this standalone checkout;
its `dvc repro` commands cannot be run here. These records describe their measured
configurations and do not establish results for another machine or data set.

For standalone implementation checks, run `just python::statadic::test-fresh` from the
repository root. The [validation guide](validation.md) describes additional optional
reference and qualification lanes. These tests do not reproduce the historical
benchmark or substitute for the original retained artifacts.

Benchmarks record preparation, host-to-device transfer, cold lowering/compilation
and synchronized warm repetitions separately, along with shape, dtype, actual
device, JAX/backend versions and native thread settings. Compiler-reported argument,
output and temporary allocation bytes are distinct from process resident memory
or measured device peak memory. A missing allocator report is unavailable evidence,
never a zero-memory claim. Timing distributions are evidence, not brittle
unit-test thresholds.

The measured operations include OLS, row-pairs refits, dense/sparse spatial
products and checked solves, projected MMD objectives, unequal-support transport,
checked capped QP and compiled Hotelling calculations. Synthetic numerical witnesses separately cover
SLX, fixed effects and a noncommuting-W spatial-panel profile. These operator
measurements do not qualify scalable sparse log-determinants, full likelihood
throughput or unmeasured general transport, packing and shrinkage paths. See the
[spatial/panel guide](spatial-models.md) for family-specific limits.

Additional bounded fitting measurements compare full and parameter-only linear
fits, fixed-schedule refits, fixed/iterated GMM and a full 64-unit SAR likelihood.
GMM records include failed and mixed valid/failed lanes. Isolated comparison
processes own their float32/float64 choice; the governed qualification stage
retains its caller-owned float64 contract. A single small full-likelihood
fixture does not establish scalable likelihood throughput. Recorded compiler
allocations remain distinct from allocator peaks and retained statistical state.

Bootstrap schedules and retained outputs occupy memory proportional to requested
draws. The shared chunk executor bounds concurrent per-refit workspaces; it does
not make total retained storage independent of the number of draws. The benchmark
varies chunk size with one fixed schedule, preserving paired draw identity.

Larger fixed-shape CPU measurements also exercise unequal supports and objective
gradients. Dense packed projected-MMD gradients at 128 versus 511 points in eight
dimensions required approximately 314 MB of compiler temporaries in float64
(157 MB in float32). This is a workspace limitation of that measured path;
unequal-support correctness and transformation support do not establish efficient
memory use at arbitrary cloud sizes. Compiler estimates are not allocator peaks.

Hotelling is an application consumer, not another STATADIC model family. Its
economic equations, certificates and equilibrium algorithms stay app-local;
the benchmark exercises their compiled numerical state and composition with
STATADIC. Certified application entrypoints own float64. The new Threefry innovation
schedule is a separate scientific change, not a numerical equivalence claim.

No GPU measurements, general resampling validity, all-family certification or
W13 release qualification are implied. For nonlinear fitted-solution derivatives,
an optimizer adjoint additionally needs a valid stationary condition and regularity;
[Optimistix documents that distinction](https://docs.kidger.site/optimistix/api/adjoints/).
