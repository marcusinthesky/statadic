# Linear fitting and parameter inference

The [source-linked OLS contrast proof](formal/claims/STATADIC-LINEAR-005.md) formalizes a finite-law, full-rank specialization of Gauss–Markov. It establishes no WLS or numerical QR implementation guarantee.

`statadic.api.regression.Regression` specifies linear, distributional, spatial,
panel and supported survival models. Predictor terms stay at the top level;
laws, observation mechanisms, structure and numerical estimators are separate
choices. Fitting returns a family-specific result with
supported `predict`, `infer(covariance=...)` and `report()` operations.
Point fitting retains state; covariance and reference distributions are selected
afterward.

```python
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression

fit = Regression(response="outcome", terms=Intercept() + Linear("exposure")).estimate(
    training
)
predictions = fit.predict(held_out)
print(fit.infer(covariance=HC("HC3")).report(reference="normal"))
```

Host joint tests name their restrictions explicitly:

```python
from statadic.api.hypotheses import Hypothesis, LinearRestriction

test = fit.infer(covariance=HC("HC3")).test(
    Hypothesis(LinearRestriction.zero("exposure")), reference="chi2"
)
print(test.report().to_text())
```

The [hypothesis guide](reference/hypotheses.md) explains fitted-name binding,
availability, F references and report metadata. Native array restrictions below
remain available for numerical composition.

The rest of this page documents the problem-plus-brick tier beneath those
classes, in the `estimation.linear`, `inference.linear_covariance` and
`inference.parameters` families. A `LinearProblem` is a registered pytree and
`LeastSquares` is a static estimator brick, so
`jax.vmap(lambda p: LeastSquares().estimate(p))` works inside `jit`/`vmap`, where
host tables cannot enter. Dyadic
specifications, point fitting and bootstrap inference have separate owners,
described below.

<!-- statadic:math STATADIC-LINEAR-004 -->

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

The [OLS probability-law result](formal/claims/STATADIC-LINEAR-010.md) requires
mean-zero errors and spherical second moments; [diagonal WLS](formal/claims/STATADIC-LINEAR-011.md)
uses fixed relative precisions. [HC0 consistency](formal/claims/STATADIC-LINEAR-012.md)
has separate limiting conditions. Each claim exposes its source locator and
human correspondence-review status; Lean checking does not verify floating-point QR.

Compare the {download}`matched statsmodels definitions <../tests/estimation/test_linear_reference.py>`
and the exact {ref}`OLS/WLS study variants <research-procedure-linear.ols-wls>`
and {ref}`covariance variants <research-procedure-linear.covariance>`.
Registration is not completed Monte Carlo evidence. Check
[capabilities](capabilities.md) and [known limits](validation/known-limitations.md)
for the requested procedure and configuration.

</details>

## Problem and brick: fit one complete sample

```python
import jax.numpy as jnp
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.parameters import coefficient_intervals

x = jnp.column_stack((jnp.ones(8), jnp.arange(8.0)))
y = jnp.array([1.0, 2.4, 2.8, 4.2, 4.9, 5.5, 7.3, 7.7])
names = ("intercept", "exposure")
fit = LeastSquares().estimate(LinearProblem(x, y, parameter_names=names))
assert bool(fit.valid)
prediction = fit.predict(x, parameter_names=names)

uncertainty = classical_covariance(fit)
intervals = coefficient_intervals(
    uncertainty, reference="t", degrees_of_freedom=fit.df_resid
)
assert bool(intervals.valid)
print(fit.parameters, intervals.estimate.se, intervals.estimate.ci)

weighted_fit = LeastSquares().estimate(
    LinearProblem(x, y, jnp.arange(1.0, 9.0), parameter_names=names)
)
```

An intercept must be a supplied design column. Inputs require a nonempty
`(observations, parameters)` design, matching response vector, complete finite
rows and full column rank. There is no automatic centering, row deletion or
column cleaning. WLS takes strictly positive **relative precision**, meaning
`Var(error_i | X) = sigma_squared / weight_i`. The common scale is estimated as
`sum(weight_i * residual_i**2) / (n-p)`. Frequency, survey and known absolute
variance weights require different contracts and are unsupported here.

The fit retains each supplied array's dtype. Its numerical working arrays use
ordinary JAX promotion across design, response and weights before whitening;
the same working precision is used for weighted scores and covariance inputs.
This follows the caller's x64 setting and does not enable a precision mode.

The frozen `LinearFit` owns immutable copies of design, response and raw weights,
parameter order, fitted values/residuals, unweighted and weighted SSE, QR maps,
scores and the aggregate mean Jacobian at its reported parameter. Scores are
`weight_i * x_i * residual_i`; the Jacobian is `-X.T @ diag(weight) @ X / n`.
No derivative panel or retained-row mask is needed. Common positive weight
rescaling preserves coefficients and estimated-scale covariance; the reported
common scale itself rescales.

Named fits require the exact ordered names for prediction and restrictions.
Unnamed fits use positional columns with width checks. Names describe a shared
column schema, not a dataset identity. Do not manually rebuild or replace fit
fields to combine state from different samples.

Shape/schema errors raise before tracing. Invalid values or rank failure return
unavailable parameters with per-lane flags and retained solver diagnostics.
Finite inputs and a successful whitened QR solve do not guarantee that the
original-scale fitted state is representable: `fit.valid` also requires finite
parameters, predictions, residuals, scores, mean derivatives and objectives.
Optional covariance factors have their own downstream availability checks.
An identified saturated fit can have valid parameters while `df_resid == 0` and
its common scale is unavailable. No pseudoinverse result authorizes ordinary
inference about unidentified parameters.

## Diagnose residuals without refitting

`result.diagnose(JarqueBera(), BreuschPagan(), RESET(), Influence())` tests
selected assumptions against the retained fit before you choose a covariance.
A robust covariance is not a diagnostic. See
[residual diagnostics](linear-diagnostics.md) for each null hypothesis,
reference law and its evidence.

## Choose covariance without refitting

All providers return `LinearUncertainty`, which owns the actual fit and its
`ParameterCovariance`. They select no reference law or reference degrees of
freedom. Returned matrices are **parameter sampling covariance**, with count and
finite-sample correction recorded separately.

| Provider | Convention |
| --- | --- |
| `classical_covariance(fit)` | Independent rows with the relative-precision variance model; estimated common scale, including `n/(n-p)` correction relative to SSE/n |
| `hc_covariance(fit, kind="HC0")` | Sum of fitted row-influence outer products; no correction |
| `hc_covariance(fit, kind="HC1")` | HC0 multiplied by `n/(n-p)` |
| `hc_covariance(fit, kind="HC2")` | Divide each row outer product by `1-h_i` |
| `hc_covariance(fit, kind="HC3")` | Divide each row outer product by `(1-h_i)**2` |
| `hac_covariance(fit, bandwidth)` | Existing uncentered Bartlett HAC in retained row/time order, explicit static `0 <= bandwidth < n` |
| `cluster_covariance(fit, groups, n_clusters=G)` | One-way cluster-summed influences; contiguous integer labels `0,...,G-1`, every group occupied, at least two groups |

Weighted leverage comes from the whitened design `sqrt(weight) * X`. Stable QR
coefficient maps supply influence rows; covariance avoids squaring conditioning
again through the generic `J.T @ W @ J` composer. The shared equation carriers
and sampling-covariance normalization remain the contracts.

Each provider accepts an additional positive `finite_sample_correction`, default
one. The recorded multiplier includes any built-in HC1/classical correction.
For the retained Paper 6 convention, pass `G/(G-1)` to cluster covariance:
this matches `statsmodels.cov_cluster(use_correction=False) * G/(G-1)` and adds
no `(n-1)/(n-p)` factor. Group labels must follow the fit's retained row order.
The contribution count remains `n`; it is never replaced by a guessed effective
sample size or the cluster count.

Host `HAC(bandwidth=..., order_by="time")` requires a finite, unique numeric
key; `order_by=None` explicitly acknowledges retained order. Native
`hac_covariance(..., row_order=indices)` reorders complete influence rows while
retaining the original fit. Neither choice verifies calendar spacing.

## Select a reference and restriction

```python
from statadic.inference.linear_covariance import hc_covariance
from statadic.inference.parameters import linear_restriction_test

robust = hc_covariance(fit, kind="HC1")
test = linear_restriction_test(
    robust,
    jnp.array([[0.0, 1.0]]),
    0.0,
    parameter_names=names,
    reference="chi2",
)
assert bool(test.valid)
print(test.test.statistic, test.test.pvalue)
```

Restrictions are `R @ beta = r`. Both `R` and `R @ covariance @ R.T` must have
full row rank; redundant rows and singular restriction variances remain
unavailable. `reference="chi2"` reports the Wald quadratic with `rank(R)` degrees
of freedom. `reference="f"` divides it by `rank(R)` and requires an explicit
positive denominator `degrees_of_freedom`.

Coefficient intervals are marginal symmetric intervals. Select `reference="normal"`
or `reference="t"`; Student intervals require explicit degrees of freedom. Normal
and chi-square references accept no degrees-of-freedom override. Classical exact
t/F calibration requires its model assumptions; robust covariance plus a chosen
t/F law is an explicit approximation, not a guarantee supplied by a method name.
Student critical values use maintained Optimistix bisection on JAX special
functions and report an unavailable result if inversion fails.

Float32 Student/F evaluation is qualified through 10,000 degrees of freedom
(both F degrees). Tests at 95%/99% confidence and nominal 5% F tails allow at most
0.05 percentage point absolute coverage/tail error, with a separate decision-margin
witness. Other finite-reference configurations are still attempted and expose
`finite_reference_outside_validated_range` independently of numerical validity.
The flag also covers degrees above 10,000 in float64 and unqualified lower
precisions. In particular, float32 beta-function error can become material at
large degrees even when the computed value is finite. Convergence and probability
bounds do not establish accuracy. A caller can choose another precision or
explicitly select an asymptotic normal/chi-square reference; the library makes
neither choice automatically. Failed inversion retains numerical failure status.

## Read availability separately from convergence

Inspect `fit.solver`, `fit.input_valid`, `fit.identified` and `fit.valid` for point
fitting. Inspect uncertainty/test/interval `.valid` and `.status` for the requested
inference. `InferenceStatus` distinguishes invalid fits, insufficient residual
degrees of freedom, numerical exact fits, invalid covariance/restrictions/reference,
numerical failure and too few clusters.

Residuals below the provider's documented dtype-relative, rowwise prediction-error
floor count as numerical exact fits. HC2/HC3 require valid leverage strictly below
one. Covariance requires positive marginal variances; joint tests add their own
restriction-rank check. These checks establish numerical availability under
declared assumptions, not exogeneity, correct specification or coverage.

Fixed-shape operations support JIT and leading-axis vmap; names, covariance kind,
bandwidth, cluster capacity and reference names are static configuration. Arrays,
fit status and reference degrees of freedom remain dynamic. Precision is caller
owned, with float32 and float64 witnesses. Full-rank OLS coefficient derivatives
with respect to design and outcome are checked away from rank changes. WLS and
inference do not inherit a blanket fitted-solution derivative guarantee; see
the [capability table](capabilities.md). No GPU performance promise is made.

## Dyadic specifications, point fits and bootstrap inference

`statadic.model.dyadic.DyadicSample` aligns outcomes with undirected endpoint IDs;
`DyadicModel` selects regressors and symmetric node effects. For a point-only
result, call `statadic.estimation.dyadic.DyadicEstimator().estimate(model)`. Its actual `DyadicFit`
retains the rank-cleaned design, coefficients, fitted values, residuals and
model-fit diagnostics. Dynamic rank selection remains an eager, caller-x64
boundary.

Bootstrap controls belong to `statadic.inference.dyadic.DyadicInference`, including
a caller-owned `NodeCountSchedule` when paired node draws are needed.
Bind them with `fitted.infer(resampling=DyadicInference(...))`. Its retained
node-bootstrap report supports
`.test(Hypothesis(LinearRestriction.zero("regressor")))` for one regressor
coefficient at zero. This preserves the native two-sided add-one sign tail and
complete draw evidence; failed draws withhold the calibrated p-value. Joint
restrictions, nonzero targets and null-imposed bootstrap tests require separate
procedures.
`Regression(..., structure=Dyads(...))` composes point fitting with the
node-bootstrap inference report. A node-count schedule induces dyad weights
through endpoint-count products; it is distinct from the row schedules in the
[resampling workflow](resampling.md). A request is not an actual fitted-state
carrier, and enabling node effects is a model choice rather than a bootstrap
option.

For entity/time and spatial structure within the same Regression interface, use the
[spatial/panel workflow](spatial-models.md).
