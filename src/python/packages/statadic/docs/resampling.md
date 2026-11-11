# Scheduled resampling and refits (preview)

The [finite full-group level proof](formal/claims/STATADIC-RANDOMIZATION-001.md) follows a published invariance-and-counting argument. Randomly sampled permutations, bootstrap coverage and add-one calibration require their own results.

`statadic.inference.resampling` separates a row-index schedule, evaluation, retained
draws and calibration. The same schedule supplies paired draw IDs across
statistics. Its executor reuses the existing chunked `lax.map`; it does not
establish exchangeability, weak dependence or any other bootstrap assumption.

```python
import jax
import jax.numpy as jnp
from statadic.inference.resampling.schedule import (
    ReplicateValue,
    evaluate_replicates,
    iid_row_schedule,
    percentile_summary,
)

data = jnp.array([1.0, 3.0, -2.0, 8.0])
schedule = iid_row_schedule(jax.random.key(42), data.shape[0], 199)


def statistic(indices):
    sample = data[indices]
    return ReplicateValue(
        values=jnp.array([jnp.mean(sample), jnp.mean(sample**2)]),
        valid=jnp.ones((), dtype=bool),
        status=jnp.array(0, dtype=jnp.int32),
    )


draws = evaluate_replicates(schedule, statistic, chunk=16)
summary = percentile_summary(draws, confidence_level=0.95)
```

Statistics may return real floating scalar or vector JAX arrays, or nonempty
nested tuples and string-key dictionaries containing those leaves. Lists,
custom objects, complex arrays and matrix-valued leaves are deliberately outside
this initial contract. Output shapes and the container structure stay fixed
across draws. Nonfinite values invalidate their draw even when the callback
reports success.

`ReplicateResult` retains every draw, its ID, validity and status, together with
requested, valid and failed counts. Raw failed values remain inspectable.
`percentile_summary` requires at least two draws and zero failures; otherwise
its bounds and SE are NaN with an explicit status. It does not drop failed
refits or report an interval conditional on success. Quantiles use linear
interpolation, and SE uses sample variance with `ddof=1`. Vector bounds are
componentwise **marginal** intervals, without a simultaneous-coverage promise.
Studentization and BCa calibration are separate methods.

## Schedules and alignment

`row_resample_schedule(indices, n_observations=..., draw_ids=...)` accepts a
supplied integer `(draws, observations)` array. Draws retain exactly the source
sample size. IDs default to draw order; supplied IDs must be unique nonnegative
integers. Shape and dtype errors raise. Out-of-range row indices invalidate
their own draw; duplicate IDs invalidate the complete paired schedule. This
remains explicit under JIT, so JAX's gather clipping cannot turn a malformed
schedule into a successful draw.

Host integer inputs must fit the caller's active JAX signed integer width;
unrepresentable NumPy row indices or draw IDs raise before conversion can wrap
their values. This boundary never enables x64. Arrays already created by JAX
carry the caller's chosen representation; information lost before entry cannot
be recovered. Dynamic schedule arrays use signed integers.

`iid_row_schedule` uses the existing split/with-replacement-choice stream.
`stationary_row_schedule` uses the existing split order, geometric restarts and
circular continuation. Finite expected block lengths below one retain the
historical clamp to one; nonfinite lengths invalidate all draws. The stationary
schedule's block-length dtype controls its transition probability. The existing
scalar adapter first casts it to the series dtype, preserving its old rule.
Both legacy `uint32[2]` and typed JAX keys are accepted. Tests freeze Threefry
streams; equal seeds across different PRNG implementations do not promise equal
streams or independent vmapped draws beyond JAX's own guarantees.

These are **row indices**, not node multiplicities, sampled cluster IDs or wild
bootstrap multipliers. A row-index carrier is not an interchangeable design for
those representations. Existing dyadic and Mantel APIs retain their own designs.
The linear adapter rejects a schedule whose observation axis differs from the
retained fitted sample; generic closures must own any additional entity/row
alignment validation.

## Frozen model specification and refitting

```python
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.resampling.schedule import linear_pairs_replicates

x = jnp.column_stack([jnp.ones(4), jnp.arange(4.0)])
y = jnp.array([2.0, 3.0, 4.0, 7.0])
fit = LeastSquares().estimate(LinearProblem(x, y, parameter_names=("intercept", "slope")))
parameter_draws = linear_pairs_replicates(fit, schedule, chunk=16)
parameter_summary = percentile_summary(parameter_draws)
print(parameter_draws.valid_count, parameter_summary.valid)
```

This tiny four-row example includes a rank-deficient resample under the tested
Threefry schedule. Its strict parameter summary is therefore unavailable, with
NaN bounds and SE; the failed draw and its identity remain in `parameter_draws`.
Inspect counts and availability before using a summary. The earlier vector
statistic example has no model-rank requirement.

`linear_pairs_replicates` gathers design, response and analytic relative
precisions with one common row schedule and retains the fitted column order.
Preprocessing and explicit design columns are frozen; no imputation, selection,
column cleaning or nuisance tuning is repeated. OLS has unit relative
precisions and uses the same weighted QR solve. Rank-failed draws remain failed
even when the numerical solver returned an iterate. A scientifically invalid
original fit cannot yield an available refit summary.

`refit_replicates(schedule, refit_fn)` supports another fitted model through a
small structural result interface: `.parameters`, `.valid`, and `.solver`.
The closure receives scheduled indices and owns its actual aligned inputs and
specification. It must state which preprocessing, nuisance fitting, selection
and tuning steps are frozen or repeated. Solver failure and invalid fit state
have distinct replicate reasons. Shape/programming exceptions propagate;
the executor does not convert arbitrary exceptions into numerical failure.
This adapter requires both scientific fit validity and solver convergence. In
particular, an affine GMM fit may expose a valid finite-budget criterion while
its iterated weight sequence has not converged; that fit remains an unavailable
replicate under this strict refit policy.

## Transformations, memory and compatibility

Factories and evaluation support eager execution, JIT and outer `vmap` at fixed
shapes. Callbacks, observation/draw counts, chunk size, confidence level and
container shapes are static during tracing. Map over an outer grid before
computing a summary for each cell; a result already batched by an outer `vmap`
does not change which axis the unbatched summary reduces. No fitted-solution
derivative or GPU performance promise follows from these transformations.

Schedules retain `O(draws × observations)` indices, and results retain all
requested output leaves and evidence. Evaluation workspace follows the chosen
chunk, including a final remainder slab; it does not batch every refit's
intermediate workspace at once. Retained schedule/output storage is separate
from that working-memory bound.

`bootstrap_estimate`, `bootstrap_confidence_interval`,
`stationary_bootstrap_estimate` and `stationary_bootstrap_ci` now reuse scheduled
evaluation. Their scalar return contracts, seed/key derivations and percentile
conventions remain intact. The historical one-draw scalar rule uses `ddof=0`
and SE zero, while the new strict summary reports insufficient draws. Legacy
`EstimateResult.converged` retains its historical scalar meaning; callers needing
per-draw failure evidence should use the preview replicate APIs.

<!-- statadic:math STATADIC-RANDOMIZATION-001 -->

## Permutation calibration and existing failure policies

`permutation_test_kernel` assembles a supplied observed statistic and null
distribution using the randomized add-one rule `(1 + b) / (1 + B)`. Ties count
as extreme: `greater` uses `null >= observed`, `less` uses `null <= observed`,
and `two-sided` uses `abs(null) >= abs(observed)`. This last rule differs from
doubling the smaller one-sided probability for an asymmetric null.

Supplying an exhaustively enumerated null does not switch this kernel to an
exact test. Exact enumeration gives an independent distribution witness; its
`b / B` exact probability and the kernel's randomized add-one probability
remain different quantities. Mantel permutations relabel nodes jointly on
both matrix axes, preserving edges' shared endpoints.

The group a randomization test draws from is a design decision. STATADIC's
level results above hold for any finite group whose action preserves the data
law. `PermutationPlan(unit=..., blocks=...)` names that group for group
comparisons and Mantel tests: whole assignment units move together, and labels
never cross exchangeability blocks. Invariance under the design's group implies
invariance under its subgroups, so declaring a subgroup keeps the level with
fewer arrangements. Declaring a larger group, such as free rows for repeated
visits of randomized subjects, can overstate significance.

Failure policies belong to their actual entry points. The preview
`percentile_summary` retains every draw and rejects a summary with any failed
draw. The existing `add_one_p` primitive likewise discards no draws: a
non-finite observation or null draw, or an empty reference set, returns NaN.
The existing eager Mantel bootstrap adapters retain
their historical conditional-on-valid-draw summaries: their kernel exposes a
validity mask, while the eager adapters discard draws with too few distinct
endpoint pairs or constant inputs. `paired_mantel_bootstrap.gap_n_boot` counts
retained draws, and both correlation arms must be valid in the same draw.
These policies are separate contracts; agreement of fixed-draw arithmetic does
not establish size or coverage under a data-generating process.

The Politis–White/Patton selector also has explicit finite-sample conventions:
STATADIC uses a `2 / sqrt(T)` autocorrelation threshold, its documented lag cap and
a minimum length of one. These differ from `arch`'s selector threshold, search
normalization and length bounds. A comparison must match those choices;
agreement on stationary/circular variance constants alone does not establish
selector parity. The first-lag heuristic remains a separate estimator.

## Finite-sample stationary mean intervals

The automatic stationary-bootstrap percentile interval has no general
finite-sample coverage guarantee. In the 10,000-trial studies with `T=256`, 999
draws and nominal 95% intervals, Gaussian IID mean coverage was 94.28%
(inconclusive under the prespecified qualification criterion), and stationary
Gaussian AR(1) mean coverage at coefficient 0.3 was 91.41% (excess error). Both
cells had zero unavailable trials. Correct schedule and percentile arithmetic
do not qualify these intervals for either cell.

The conditional variance of a stationary-bootstrap sample mean can be evaluated
exactly without drawing bootstrap samples. For centered observations `z`, let
`gamma_c(h) = sum_i z[i] * z[(i+h) % T] / T` and `q = 1 - 1/b`, where `b >= 1`
is the expected block length. Uniform restarts and circular continuation imply

```text
Var*(mean*) = [gamma_c(0)
              + 2 sum_{h=1}^{T-1} (1-h/T) q^h gamma_c(h)] / T.
```

For Gaussian observations with covariance `Sigma`, replace each circular lag
covariance by its expectation under `M Sigma M`, where
`M = I - ones(T,T)/T`. This gives the expected conditional variance, which can
be compared with the true finite-sample mean variance `sum(Sigma)/T^2`.
The [independent witnesses](../tests/inference/resampling/test_schedule_stationary_variance_reference.py)
verify both identities against all 256 length-four Markov paths and their
Gaussian count-contrast quadratic forms.

For stationary Gaussian AR(1) data with coefficient 0.3 and innovation variance
one, `T=256` and population MSE block length approximately 4.81009, the expected
bootstrap mean variance is approximately 0.839846 times the true mean variance.
Thus finite-sample variance attenuation remains even when block-length
ingredients are known from the population. Increasing the draw count reduces
quantile Monte Carlo error; it does not remove this attenuation. A variance
ratio alone does not prove a coverage probability.

The [held-out paired diagnostics](../tests/qualification/test_stationary_diagnostics.py)
compare the current and arch selectors and fixed lengths 1, 4, 8 and 16 on
1,000 new datasets per original DGP, retaining `T=256`, 999 draws and the
percentile method. They share data and restart randomness within each trial
and retain failures, lengths, variances, endpoints and paired coverage
differences. These are mechanism diagnostics with no qualification gate or
selection of a replacement default.
