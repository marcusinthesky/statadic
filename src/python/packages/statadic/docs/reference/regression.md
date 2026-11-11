# Regression: familiar constructors and explicit components

Named constructors return the same {py:class}`~statadic.api.regression.Regression`
specifications you can write
explicitly. They select a model and its native estimator; estimation, diagnostics
and inference remain separate decisions.

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.estimation.linear import LeastSquares

terms = Intercept() + Linear("x")
model = Regression.ols("y", terms)
explicit = Regression("y", terms, estimator=LeastSquares())
# Estimate when data supplies the named x and y columns.
```

| Constructor | Explicit choice | Additional inputs |
| --- | --- | --- |
| {py:meth}`Regression.ols <statadic.api.regression.Regression.ols>` | `LeastSquares()` | Unweighted observations |
| {py:meth}`Regression.wls <statadic.api.regression.Regression.wls>` | Least squares with relative precision | Required `precision=RelativePrecision("w")` |
| {py:meth}`Regression.logit <statadic.api.regression.Regression.logit>` | `Binomial(link="logit")` | Optional offset and matching estimator |
| {py:meth}`Regression.probit <statadic.api.regression.Regression.probit>` | `Binomial(link="probit")` | Optional offset and matching estimator |
| {py:meth}`Regression.poisson <statadic.api.regression.Regression.poisson>` | `Poisson()` | Optional offset, exposure and matching estimator |
| {py:meth}`Regression.aft <statadic.api.regression.Regression.aft>` | Censored response and AFT law | Required `RightCensored(...)`, `law=...`; optional matching estimator |

All constructors require explicit terms, including an intercept when wanted.
For grouped binomial outcomes use `Observed("successes", trials="trials")` as
the response. AFT keeps censoring and delayed entry on `RightCensored` and requires
an explicit Weibull, lognormal or log-logistic law. Numerical options belong to
the corresponding native estimator brick.

Use explicit `Regression(...)` composition for panel, spatial, IV, dyadic and
geometric models. An Ordination may be a predictor term, retaining its training
transformation. The [composition guide](composition.md) develops those workflows.

The [reference index](../validation/references.md) lists exact matched
configurations and their evidence. A familiar name alone promises neither a
Gaussian error law for least squares nor a calibrated inferential procedure.
Learn the full workflow in the [OLS chapter](../tutorials/ols/index.md).

## OLS and WLS retained state

### Host model

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

Regression(response="y", terms=Intercept() + Linear("x"))
```

This is the named-observation interface. Its default estimator brick is `LeastSquares`. Fitting returns a `LinearResult` whose `.core` is the actual numerical `LinearFit`, with retained design and observation context.

| Operation | Contract |
|---|---|
| `model.estimate(data)` | Prepare and fit aligned observations; produce new retained state |
| `model.prepare(data)` / `estimate_prepared(...)` | Prepared state must belong to the producing model instance |
| `result.predict(data)` | Apply the retained design and exact parameter schema |
| `result.infer(covariance=...)` | Attach a supported family-specific covariance to this actual fit, without refitting |
| `result.report()` | Point-fit information in a [Report](reporting.md); does not select inferential reference laws |
| `result.save_predictor(path)` | Restricted prediction artifact; do not promise reconstructed inference |

### Native numerical door

`LeastSquares().estimate(LinearProblem(design, response, ...))` takes complete arrays. Supply the intercept column explicitly. Native named prediction requires the same ordered parameter names; unnamed fits use positional columns with width checks.

The fit requires supported finite input and full column rank. Shape/schema errors raise; invalid numerical lanes and rank failures expose unavailable results and diagnostics. There is no automatic row deletion, centering, column cleaning or license to interpret an unidentified pseudoinverse solution as ordinary inference.

### WLS is relative precision

The WLS contract uses strictly positive relative precisions, with conditional error variance `sigma² / weight_i`. It is not a frequency-weight or survey-weight interface. The estimated common scale uses the actual weighted residuals and residual degrees of freedom. The fit retains the arrays, their effective numerical precision, parameter order, residuals and score/Jacobian information needed for downstream covariance.

### Inference remains another step

Covariance providers return sampling covariance for the actual fit. Reference laws and required degrees of freedom are separate. Check interval or test validity/status independently of point-fit validity. See [inference choices](../guides/inference.md) and [the OLS inference lesson](../tutorials/ols/infer.md).

<details><summary>Assumptions and evidence</summary>

The [fixed-design unbiasedness result](../formal/claims/STATADIC-LINEAR-009.md)
assumes the stated first moments; the
[Gauss–Markov variance comparison](../formal/claims/STATADIC-LINEAR-010.md)
additionally assumes spherical second moments and full rank. Neither verifies
the QR/JAX implementation or interval coverage. WLS has its own
[known-diagonal-precision assumptions](../formal/claims/STATADIC-LINEAR-011.md).
Claim pages retain exact source and review status.

The {download}`precision reference controls <../../tests/testing/test_precision_references.py>`
contain matched OLS/WLS witnesses with their explicit tolerances and scope.

Inspect [matched reference configurations](../validation/references.md),
[registered studies and selected reports](../validation/research.md) and
[known limits](../validation/known-limitations.md) separately. A constructor or
passing numerical comparison does not qualify every composition.

</details>
