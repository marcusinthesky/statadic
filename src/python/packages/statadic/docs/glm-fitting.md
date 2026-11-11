# Binomial-logit and Poisson-log fitting

For Gamma/log-mean, negative-binomial NB2 and binomial-probit workflows, see
[additional likelihood regression](reference/extended-regression.md). Those
families retain their own dispersion and likelihood contracts; the canonical
fixed-dispersion fitting below is unchanged.

The public workflow is `statadic.api.regression.Regression`: combine an `Observed`
response and top-level predictor terms with `law=Poisson()` or
`law=Binomial(link="logit")`. Estimation returns a
`PoissonResult` or `LogisticResult` with `predict(kind=...)`, `infer` and
`report`, as in the [conditional-law reference](reference/conditional-laws.md).
This page documents
the problem-plus-brick tier beneath them, for use inside `jit`/`vmap`: the
pytree problems `PoissonProblem`/`BinomialProblem` and the static estimator
bricks `PoissonMLE`/`BinomialLogitMLE`, which carry `max_steps` and `tol`.

These problems and bricks extend the same fitted-state, mean-equation, uncertainty and
resampling composition used by linear models. Inputs are complete finite arrays
with explicit columns, including any intercept. Rows and columns are retained in
their original order; no cleaning or rank reduction is implicit.

```python
import jax.numpy as jnp
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    PoissonMLE,
    PoissonProblem,
)
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.parameters import coefficient_intervals

x = jnp.column_stack((jnp.ones(6), jnp.arange(6.0)))
fit = BinomialLogitMLE().estimate(
    BinomialProblem(x, jnp.array([2, 4, 3, 6, 7, 8]), trials=10)
)
uncertainty = glm_covariance(fit, method="model")
interval = coefficient_intervals(uncertainty, reference="normal")
probabilities = fit.predict(x)
expected_successes = fit.predict(x, trials=10, scale="mean")
counts = PoissonMLE().estimate(
    PoissonProblem(x, jnp.array([0, 1, 1, 3, 2, 5]), exposure=2.0)
)
```

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

The [binomial score](formal/claims/STATADIC-GLM-002.md) and
[Poisson score](formal/claims/STATADIC-GLM-003.md) are likelihood identities.
The [fitted-root limit](formal/claims/STATADIC-GLMCOV-003.md) additionally assumes
regularity and convergence. Claim pages retain source locators and human
correspondence-review status; the identities alone prove neither existence of
a finite fitted optimum nor calibrated intervals.

Read the {download}`matched statsmodels/glum comparisons <../tests/estimation/test_glm.py>`,
{ref}`exact canonical-GLM study mappings <research-procedure-glm.canonical>` and
{ref}`covariance mappings <research-procedure-glm.covariance>`.
Studies need retained outcomes and uncertainty, beyond registration. See
[transformation scope](capabilities.md) and [known limits](validation/known-limitations.md).

</details>

## Statistical and retained-state conventions

Binomial outcomes are integer successes, with positive integer trials supplied as
a scalar or matching row vector. The logit model gives probabilities `p`; expected
successes are `trials * p`. Poisson outcomes are nonnegative integer counts, with
log expected count `X @ beta + offset + log(exposure)`. Exposure is positive and
defaults to one; offset defaults to zero. Both can be supplied together. Exposure
is neither an outcome rate nor a frequency weight. Dispersion is fixed at one;
frequency, survey, quasi-likelihood and penalized fitting require other contracts.

`GLMFit` is an immutable registered array pytree. It owns actual design, outcome,
trials, offset, exposure, parameter order, linear predictor, means/probabilities,
residuals, full log likelihood and deviance. Likelihood includes binomial
coefficient or Poisson factorial constants; deviance uses the saturated likelihood
with zero-count limits. Poisson `probabilities` is unavailable (NaN).

One score contribution is one retained row, including a grouped-binomial row:
`score_i = x_i * (y_i - mean_i)`. The aggregate mean Jacobian is
`-X.T @ diag(variance_i) @ X / n`, where variance is `trials*p*(1-p)` or the Poisson
mean. Scores, derivatives and likelihood all use the same reported parameter.
No observation-by-parameter-by-parameter derivative panel is allocated.

`GLMFit.predict` requires aligned explicit columns. Named fits require the exact
ordered names. `scale="link"` returns the linear predictor; `"response"` returns
binomial probability or expected Poisson count; `"mean"` returns expected
successes/counts. Prediction rows supply their own trials, offset and exposure,
defaulting to one/zero/one; training rows are never silently recycled.

## Existence, rank and numerical availability

Full column rank and a finite optimum are separate requirements. Binomial
recession directions satisfy `X_i d = 0` on interior success counts,
`X_i d >= 0` on all-success rows, and `X_i d <= 0` on zero-success rows, with at
least one strict boundary margin. Poisson directions vanish on positive-count
rows and are nonpositive on zero-count rows, with at least one strict margin.
Finite offsets and positive finite exposures leave these cones unchanged. These
conditions cover complete and quasi separation; see Albert and Anderson
(1984; DOI `10.1093/biomet/71.1.1`) and
[Correia, Guimaraes and Zylkin](https://arxiv.org/abs/1903.01633).

The existing QP dependency projects the mean oriented boundary row onto the cone
after fixed-shape SVD nullspace reduction of interior rows. A checked strict
primal direction records `SEPARATED`; a zero projection with a checked positive
dual combination records `FINITE`. The candidate dual combination is projected
into the constraint nullspace and its corrected weights must exceed a bound
scaled by dtype, shape, conditioning and candidate magnitude; a small absolute
residual alone cannot certify finiteness. Unresolved evidence records `UNKNOWN` and
cannot authorize a fit. Residuals, direction and margin remain inspectable in
`fit.existence`. This is a floating-point certificate: its normalized residual
resolution is `max(1e-7, 2048*dtype_epsilon)`, and the SVD rank threshold follows
array shape and dtype. Samples too near that resolution may be unavailable.
There is no claim of exact arithmetic or a general high-dimensional separation
algorithm. All-zero Poisson with an intercept is separated; without an intercept,
the actual cone can instead admit a finite solution.

Optimistix BFGS globalizes the unpenalized likelihood after positive column scaling;
at most 64 BFGS steps are followed by Optimistix Newton score refinement with
Lineax QR whenever the total iteration budget leaves refinement steps. Backend
termination is retained rather than relabeling an exhausted BFGS iterate.
Default tolerance is `max(1e-8, 32*dtype_epsilon)` with 256 steps. Solver status,
input validity, full-column-rank identification and fit validity are separate.
Invalid inputs, unresolved/separated cones, numerical nonconvergence or
unrepresentable scientific state yield NaN scientific outputs. Original inputs
and numerical diagnostics remain available. No pseudoinverse fit authorizes
ordinary inference.

## Covariance, calibration and refitting

`glm_covariance` returns a fit-attached `GLMUncertainty` through the shared
`parameter_covariance` and aggregate-Jacobian sandwich. `method="model"` uses
the information equality at fixed dispersion one. HC0 uses uncentered fitted
score outer products with denominator `n`; HC1 multiplies HC0 by `n/(n-p)` and
requires positive residual degrees of freedom. An additional positive
`finite_sample_correction` is recorded separately. The returned matrix is
parameter sampling covariance, after the common division by retained row count.

Choose normal/t intervals and Wald reference distributions separately through
the shared parameter-inference API. Robust covariance does not establish a
likelihood-ratio reference or justify a dispersion model. Invalid fits cannot
yield available covariance or ordinary inference.

The generic `refit_replicates` executor can refit either family from a declared
row schedule; the callable must resample outcome, design, trials, offsets and
exposure together. Failed separated or nonconverged replicates retain their draw
identity and invalidity instead of being silently dropped.

Fixed-shape fits, covariance and prediction support JIT and leading-axis batching
under caller-owned float32 or float64 precision. Static family, parameter names
and iteration budget describe the schema or algorithm. Likelihood derivatives
are distinct from derivatives of the fitted optimizer solution: no fitted-solution
derivative or accelerator-performance promise is made here.
