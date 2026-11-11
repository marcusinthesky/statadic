# Numerical and host result boundaries

JAX transforms cross one numerical boundary: array-like artifacts are packed by
an eager entry point, numerical code accepts JAX arrays and PRNG keys, and the
result is a registered array pytree. Python scalars and legacy report objects are
created only by a named host adapter after that pytree leaves `jit` or `vmap`.

```text
ArrayLike + seed -> eager validation/packing -> Array + PRNGKey
                  -> transformable core -> TestResult / EstimateResult
                  -> named to_host_* adapter -> legacy/report object
```

## Public result inventory

Plain mathematical functions that return one array/scalar value, such as
`energy_distance`, `cdist` and `sharpe_ratio`, are numerical values rather than
result-assembly boundaries. Public families may combine eager preparation with
fixed-shape numerical kernels; a result record alone does not promise JIT.
Paths below are relative to `statadic`.

| Public boundary | Result today | Execution contract | Owner and limits |
|---|---|---|---|
| `core.results.TestResult`, `EstimateResult` | Registered array pytrees and static resample counts | Transformable primary numerical contracts | Procedure-specific assumptions remain with their producers |
| `core.matrices.DMat` and branded constructors | Registered array carrier with law/provenance metadata | Transformable mathematical carrier | Checked value domains and permitted constructors are distinct |
| `discrepancy.mmd.energy.empirical_mean_ground_distances` | `PairwiseEstimateMatrix` of empirical V-statistic ground means | Eager cloud validation/packing; fixed-shape stacked/packed kernels support JIT/vmap | True counts and self-pairs; no distance-law claim |
| `operators.design.fwl_residualize` / `fwl_coefficient` | `FWLResiduals` array pytree / Python float | Fixed-shape residualization supports JIT/vmap; coefficient wrapper is eager | Caller precision and rank-cleaned columns |
| `inference.resampling.permutation.permutation_test_result`, `permutation_test_kernel` | `TestResult` | Primary key-taking JAX paths support JIT/vmap | Caller supplies the statistic and exchangeability assumptions |
| `inference.resampling.permutation.permutation_test_result` | `TestResult`, reported as `HypothesisTest` | Eager report via `inference.resampling.results.to_host_test_result` | Explicit host conversion after numerical computation |
| `inference.resampling.bootstrap.bootstrap_estimate` | `EstimateResult` | Key-taking JAX path supports JIT/vmap | Bootstrap law remains a caller assumption |
| `inference.resampling.bootstrap.bootstrap_confidence_interval` | Two Python floats | Eager adapter via `inference.resampling.results.to_host_confidence_interval` | Scalar compatibility boundary |
| `inference.resampling.block_length` selectors and stationary CI | Host scalars/CI; separate numerical primitives | Eager selectors and report adapters; stationary evaluation shares chunked execution | Precision and dependence conventions remain explicit |
| `estimation.linear`, `estimation.gmm`, `estimation.glm` fitting doors | Immutable actual-fit pytrees with solver/identification evidence | Fixed-shape JIT/vmap for documented doors | [Capabilities](capabilities.md) distinguishes objective and fitted-solution derivatives |
| `inference.linear_covariance`, `inference.gmm.covariance`, `inference.glm_covariance`, `inference.parameters` | Fit-attached uncertainty and calibrated array pytrees | Fixed-shape JIT/vmap | Actual sample, corrections, reference and degrees of freedom stay separate |
| `inference.resampling.schedule` | Schedules, replicate evidence and strict marginal summaries | Fixed-shape JIT/vmap; shared chunked execution retains all failures and IDs | [Schedules and refits](resampling.md) |
| `estimation.panel`, `inference.panel_covariance` | Actual static-panel fits and attached uncertainty | Eager index/sample preparation; documented fixed-shape numerical operations | General panels can be unbalanced; [panel contracts](spatial-models.md) |
| `estimation.spatial`, `inference.spatial` | Actual spatial fits, covariance, impacts and diagnostics | Explicit graph preparation; documented fixed-shape numerical doors | Dense/sparse limits and time-varying-W scope in the [spatial guide](spatial-models.md) |
| `api.association.Mantel` | `MantelResult`, `MantelInference` | Eager workflow over fixed-shape correlation/permutation kernels | Pure correlation statistics remain in `association.mantel` |
| `inference.disco.disco_permutation_null` | Resampled null draws | Numerical label-permutation null over a fixed matrix; the add-one tail is `inference.calibration.add_one_p_kernel` | Pure decomposition stays in `association.decomposition` |
| `inference.dispersion.dispersion_permutation_null` | Array of null F statistics | Fixed-shape permutation computation | Pure PERMDISP statistic stays in `association.dispersion` |
| `inference.mixture.corrected_mixture_test`, `inference.equivalence.equivalence_test` | `MixtureTestResult` | Eager composite; mixture orchestration assembles host report fields | Selection/equivalence assumptions are procedure specific |
| `inference.gmm.hypothesis` fixed-moment tests | `WaldResult`, `HansenJResult` | Eager checked reports over numerical kernels | Actual affine fitting belongs to `estimation.gmm`; `hansen_j_from_fit` consumes its retained weight and state |
| `inference.kappa` delta and Jacobian diagnostics | `KappaDeltaResult`, `MomentJacobianResult` | Eager reports over explicit numerical kernels | Reference and identification conditions remain explicit |
| `inference.gms.indicator_gms`, `inference.wolak.wolak_test` | `GmsResult`, `WolakResult` | Eager orchestration over constrained JAX operations | Solver success alone does not establish calibrated inference |
| `estimation.gel.FixedGEL.estimate`, `AffineGEL.estimate` | `FixedGelFit`, `AffineGelFit` | Eager fitting with retained multiplier, frozen basis and separate solve evidence | Private fixed-shape dual kernels are transformable; public rank/validation doors are host boundaries |
| `inference.gel.gel_test` | `GelResult`, `AffineGelResult` | Eager statistic/reference selection from an actual fit | [GEL fit-then-test](gel-fitting.md); explicit iid acknowledgement, no refitting |
| `estimation.covariance` | `LedoitWolfKernelResult`, `ShrinkageResult` and covariance arrays | `Estimand(Covariance(), estimator=...)` validates eagerly; `CovarianceProblem` and numerical bricks retain the JAX boundary | Estimates covariance of observed vectors; parameter uncertainty belongs to inference |
| `model.dyadic` specifications | `DyadicSample`, `DyadicModel` | Eager endpoint-label and regressor specification | Contains no point solver or bootstrap controls |
| `estimation.dyadic.DyadicEstimator.estimate` | Actual `DyadicFit` with retained design/fitted/residual state | Eager data-dependent rank selection and checked least squares | Caller precision; no node-bootstrap uncertainty |
| `inference.dyadic` | Node schedules, `DyadicBootstrapResult`, combined `DyadicFitResult` | Eager design/report orchestration with numerical bootstrap kernels | Combined coordinator consumes `DyadicFitRequest`; node counts are not row indices |
| `inference.sharpe` tests, deflation and comparisons | Procedure-specific reports and float interval pairs | Eager reports over caller-precision scalar/bootstrap kernels | Sample ratio and moment reductions belong to `estimation.sharpe` |
| `geometry.barycentre.mmd.energy` with diagnostics | Weights plus `EnergyBarycentreDiagnostics` | Eager validated doors and separate transformable batched kernel | Generic MMD and energy share the same simplex solver |
| `discrepancy.transport.exact` JAX doors plus W1/W2 wrappers | JAX arrays/scalars | Fixed-shape Optax Hungarian core supports JIT/vmap | Eager artifact doors validate costs/unit rows and retain caller precision |
| `optimize.backends` numerical/result-bearing doors | Arrays or `SolverResult` evidence | Lineax/Optimistix numerical adapters | Backend status is distinct from statistical identification |
| `geometry.embedding`, neighbour summaries | Array tuples / typed dictionaries | Explicit numerical/artifact and eager summary boundaries | Dynamic output preparation remains separate from fixed-shape kernels |
| `sample.packed.PackedClouds`, `pack`, `checked_packed_clouds` | One empirical carrier with dynamic data, count and mass leaves | Eager checked factories; tracer-safe structural constructor; JIT/vmap on fixed capacities | Differentiate floating coordinate/logit parameter trees with counts fixed; outer observations and atoms remain separate axes |
| `discrepancy.mmd.weighted`, energy weighted and transport matrix producers | Weighted pairwise/cross arrays and qualified `DMat` | JAX arithmetic on prepared fixed-shape measures; bounded JIT/vmap witnesses | Construction laws are method-specific; materially negative/undefined lanes remain unavailable |
| `api` data, representations and fitted results | Host contexts wrapping existing numerical state | Eager preparation, identity checks and result assembly | Reports/artifacts are host boundaries; a workflow object promises no fitted-solution derivative |
| `sample.reduction.fit_projection` | `Projection` dataclass | Eager/random construction | Fit on a selection sample when later inference requires it |
| `optimize.psd` diagnostics | `RepairDiagnostics` with numerical convergence evidence | Namedtuple array leaves; no NumPy report conversion | Numerical feasibility repair selects no covariance estimand |

## Concretization ledger

Permutation and bootstrap numerical paths leave Python scalar conversion to
`to_host_test_result` and `to_host_confidence_interval`, which explicitly reject
tracers. Integer seeds likewise belong to eager adapters. These guarantees do
not extend automatically to every statistical family.

| Current location | Why host conversion remains |
|---|---|
| `inference/mixture/_split_runner.py` and `inference/mixture/contracts.py` | Procedure-specific mixture/equivalence report assembly |
| `inference/resampling/block_length.py`, `operators/{longrun,covariance}.py` | Validation and report adapters around numerical primitives |
| `inference/mantel.py`, `inference/gmm/hypothesis.py`, `inference/kappa.py`, `estimation/covariance/` | Checked result assembly and eager option/domain validation |
| `inference/{gms,wolak}.py` | Constrained-optimization orchestration and report assembly |
| `estimation/gel/{fit,_dual}.py`, `inference/gel/hypothesis.py` | Eager rank/optimizer fitting followed by separate test calibration; the private dual kernels remain numerical |
| `model/dyadic/validation.py`, `estimation/dyadic/`, `inference/dyadic/` | Label/rank decisions, actual point state and separate bootstrap reports; the fixed-shape least-squares solve delegates to Lineax |

Host/report code may convert deliberately. A numerical core must keep its
array/key contract and avoid host concretization. Qualification of a
transformable door covers direct execution, nested JIT and leading-axis vmap;
its corresponding host adapter rejects traced calls.

## Precision boundary: who owns `x64`

The caller chooses floating inputs and JAX process configuration before entering
STATADIC. The library does not request a different floating precision, mutate the
process flag, or refuse computation solely because of dtype. Mixed floating
inputs follow normal JAX promotion; untyped inputs follow JAX's native defaults.
Boolean structural columns and integer indices do not impose a floating policy.
Numerical constants derive their dtype from the operation's inputs.

Strict identity, schema and option checks belong at host preparation boundaries.
Numerical kernels retain fixed-shape validity, solver and qualification evidence
without Python truth tests on tracers. Supported backend operations are attempted
in caller precision; an unsupported backend operation remains visible rather
than triggering a hidden upcast. Finite-reference results additionally expose
`finite_reference_outside_validated_range`, independently of numerical success.
That flag records limits of the independent tests, not an alternative reference
law or an automatic precision choice.

Historical experiments explain why caller selection still matters. Removing
library-owned x64 scopes without selecting precision in pipeline clients caused
all 1,999 dyadic bootstrap draws to fail on severely ill-conditioned designs,
and Wolak solver lanes also failed. Those observations justify retained numerical
diagnostics and explicit client configuration, not a library precision override.
A current pipeline stage may deliberately select x64; another client may choose
bfloat16 or float32 and inspect the resulting evidence. Neither choice by itself
establishes numerical accuracy or statistical qualification.

**Why a door was never sufficient — measured, jax 0.10.2, CPU, caller context
x64 off.** `statadic` no longer contains such a `door`; the probe below lives in the
test file and is retained because it is the evidence that hoisting ownership to
the caller was the only correct fix, not a matter of taste. Rows are identical
whether the flag is off at import or turned on and then dropped with
`with jax.enable_x64(new_val=False)`, which is what the pinning test does:

```python
probe = np.array([1.0 + 1e-10, 1.0], dtype=np.float64)  # the gap vanishes in float32
batch = np.stack([probe, probe])


def door(x):
    with jax.enable_x64(new_val=True):
        r = jnp.asarray(x, dtype=jnp.float64)
        return r[0] - r[1]
```

| call | result | why |
|---|---|---|
| `door(probe)` | `1.000000082740371e-10` | eager entry: the context governs the values |
| `jax.jit(door)(probe)` | `0.0` | the input tracer is already `float32`; the door upcasts data that was rounded on the way in |
| `with jax.enable_x64(True): jax.jit(door)(probe)` | `1.000000082740371e-10` | caller-owned: correct |
| `jax.vmap(door)(batch)` | `[1.000000082740371e-10, ...]` | **exception:** the batching trace carries the raw host buffer, so the door still owns the conversion |
| `jax.vmap(door)(jnp.asarray(batch))` | `[0.0, 0.0]` | the caller committed the array first; canonicalization already happened |
| `jax.jit(jax.vmap(door))(batch)` / `jax.vmap(jax.jit(door))(batch)` | `[0.0, 0.0]` | a `jit` boundary in either position commits the argument |
| `jax.grad(door_sq)(probe)` | `float32` cotangent | a cotangent follows the primal input dtype — true *eagerly* too, not only under `jit` |

The `jit` result's *dtype* is `float64`, so a dtype assertion does not see this;
only the value does. `float32(1.0 + 1e-10) - float32(1.0)` is exactly `0.0`,
which is why the probe discriminates.

**Consumers that own the context** (process-global `jax.config.update`, so the
claim is checkable rather than aspirational):

| owner | site |
|---|---|
| **pipeline CLI, per stage** | each numerical command explicitly calls `pipeline.precision.own_float64` before constructing arrays; the call itself enters the command's Semflow closure |
| Paper 2 hedge stage | `apps/pipeline/.../paper2/_hedge/contracts.py:22` |
| Paper 3 empirical optimizer (retired in t112) | formerly `apps/pipeline/.../paper3/_empirical/optimize.py:37` |
| Paper 5 energy-robustness optimizer | `apps/pipeline/.../paper5/_energy_robust/optimize.py:34` |
| this package's test session | `packages/statadic/tests/conftest.py:47` |

The numerical command call sites own precision, without a second stage-membership
registry or blanket CLI callback flip. Some retained paths intentionally use
float32; a caller selecting float64 for another stage must not change them.

A caller can also own the context *scoped* rather than process-globally, which
is the same ownership claim with a narrower blast radius. Paper 1's DISCO stage
is the worked example: it wraps its whole numerical body in
`with jax.enable_x64(new_val=True)` (`apps/pipeline/.../paper1/disco.py:151`,
contracted at `:60`) and embeds through `geometry.embedding.pcoa_kernel` inside it. The kernel opens
no scope of its own, so its precision is exactly what the stage owns — as is
now true of every function in this package, including the retired eager
artifact door that used to open x64 itself.

**Spawn.** A spawned interpreter inherits no process-global JAX flag, so a
callback-time update reaches the parent only. The one stage tree that fans out
to `spawn` workers — `paper2._hedge` — owns the flag at module import so its
cold-importing workers get it too, pinned by
`apps/pipeline/tests/papers/paper2/test_hedge_spawn_runtime.py:53`.

The JAX property behind all of this is pinned by
`tests/migration/test_sharpe_jax_scipy.py::test_door_scoped_x64_does_not_survive_jit`,
which builds its own local `door` rather than calling statadic,
which enters `with jax.enable_x64(new_val=False)` — an explicit x64-disabled
caller, since `tests/conftest.py:47` has the global flag on — and asserts every
row above: the `jit` value, its `float64` dtype (a dtype assertion alone would
pass while being wrong), the `float32` cotangent, the bare-`vmap` exception, the
three committed-argument cases, and the caller-owned control. It asserts the
*un*-preserved direction on purpose. A future JAX release that changed
canonicalization order — making the door's context reach the input — would turn
that `0.0` into `1.000000082740371e-10` and fail there, rather than silently
moving a manuscript number.
