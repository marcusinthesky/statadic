---
type: reference
description: Specify, fit and draw inference from regression models, from ordinary least squares to explicit compositions.
---

# Regression

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression

data = {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.1, 1.8, 3.2, 3.7, 5.3, 5.8]}
fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
report = fit.infer(covariance=HC("HC3")).report(reference="normal")
print(report.to_text())
```

## Description

{py:class}`~statadic.api.regression.Regression` describes a model by naming its
response and its predictor terms, and optionally a response law, a structure for
the observations and a numerical estimator. Calling `estimate` fits the model to
data and returns a result that keeps the design built from the training rows.
That result can predict new rows, run diagnostics and produce inference, and
each of those is a separate request.

Named constructors such as `Regression.ols` select familiar combinations and
return the same specification you could write explicitly. For panel, spatial,
instrumental-variable, dyadic and geometric models, compose `Regression`
directly, as described in the [composition reference](composition.md).

## Options

| Constructor | Equivalent explicit choice | Additional inputs |
| --- | --- | --- |
| {py:meth}`Regression.ols <statadic.api.regression.Regression.ols>` | `LeastSquares()` | None |
| {py:meth}`Regression.wls <statadic.api.regression.Regression.wls>` | Least squares with relative precision | `precision=RelativePrecision("w")` |
| {py:meth}`Regression.logit <statadic.api.regression.Regression.logit>` | `Binomial(link="logit")` | Optional offset and estimator |
| {py:meth}`Regression.probit <statadic.api.regression.Regression.probit>` | `Binomial(link="probit")` | Optional offset and estimator |
| {py:meth}`Regression.poisson <statadic.api.regression.Regression.poisson>` | `Poisson()` | Optional offset, exposure and estimator |
| {py:meth}`Regression.aft <statadic.api.regression.Regression.aft>` | Censored response with an AFT law | `RightCensored(...)` and `law=...` |

Every constructor requires explicit terms, including the intercept when one is
wanted. Grouped binomial outcomes use `Observed("successes", trials="trials")` as
the response. The [response laws reference](conditional-laws.md) covers the
binomial, Poisson and other likelihood families.

Uncertainty is requested from a fitted result with `infer(covariance=...)`:

| Request | Use when |
| --- | --- |
| `ModelBased()` | Rows are independent with constant error variance |
| `HC("HC0")` to `HC("HC3")` | Rows are independent but the error variance may differ between them |
| `Clustered("column")` | Rows share shocks within groups named by a column |
| `HAC(bandwidth=b, order_by="time")` | Nearby observations in a stated order may be correlated |

Each request accepts a positive `finite_sample_correction`, which multiplies
the covariance and is recorded in the report. Building a report then selects
the reference distribution, with `reference="normal"` or `reference="t"` for
intervals and `"chi2"` or `"f"` for joint tests.

## Results

Fitting least squares returns a `LinearResult`. Its main operations are:

| Operation | What it does |
| --- | --- |
| `result.parameters`, `result.parameter_names` | The estimated coefficients in the order of the design |
| `result.predict(data)` | Applies the training design to new rows, which must contain the predictor columns |
| `result.diagnose(...)` | Runs residual diagnostics on the fitted model. See [linear diagnostics](linear-diagnostics.md) |
| `result.infer(covariance=...)` | Adds sampling covariance for these coefficients without refitting |
| `result.report()` | Summarises the point fit as a [report](reporting.md), without standard errors |
| `result.save_predictor(path)` | Saves what prediction needs. See [saved artifacts](artifacts.md) |

`result.valid` reports whether every fitted quantity is finite and the design
has full column rank, and `require_valid()` raises when it does not. Inference
results carry their own availability, which you can check with
`report.require_available("interval")`.

For array workflows inside `jax.jit` or `jax.vmap`, the numerical layer offers
`LeastSquares().estimate(LinearProblem(design, response))`, with covariance from
`statadic.inference.linear_covariance` and intervals and Wald tests from
`statadic.inference.parameters`. The design must include any intercept column,
and named fits require the same ordered names for prediction.

## Methods and formulas

For a design $X$ with $n$ rows and $p$ columns, ordinary least squares solves
$\hat\beta = \arg\min_\beta \lVert y - X\beta \rVert^2$ through a QR
factorisation. Weighted least squares treats each weight as a relative
precision, so that $\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2 / w_i$,
and estimates the common scale as $\sum_i w_i r_i^2 / (n - p)$.

The heteroskedasticity-consistent estimators sum the outer products of each
row's influence on $\hat\beta$. HC1 multiplies HC0 by $n/(n-p)$, HC2 divides each
term by $1 - h_i$ and HC3 by $(1 - h_i)^2$, where $h_i$ is the leverage of row
$i$. Clustered covariance sums influences within each group before forming
outer products, and HAC weights the products of nearby rows with Bartlett
weights up to the stated bandwidth.

A linear restriction $R\beta = r$ is tested with the Wald statistic, compared
with $\chi^2_{\operatorname{rank}(R)}$ or, after division by the rank, with an
$F$ distribution whose denominator degrees of freedom you supply.

<!-- statadic:math STATADIC-LINEAR-004 -->

## Assumptions and limits

- Least squares estimates a conditional mean. Causal interpretation requires an
  identification argument about how the data were generated.
- The design must have full column rank. STATADIC does not delete rows, centre
  columns or fall back to a pseudoinverse when it is rank deficient.
- Weights describe relative precision. Frequency weights and survey weights
  need different estimators.
- Normal and chi-square references are large-sample approximations. Student
  and F references are exact only under the classical model, so combining them
  with a robust covariance is a deliberate approximation.
- HC2 and HC3 require every leverage to be below one, and clustered covariance
  requires at least two groups.
- HAC orders observations by the column named in `order_by`, which must hold
  unique numbers, or by the current row order when `order_by=None`. Lags count
  adjacent observations in that order, so calendar gaps are your responsibility.
- In 32-bit precision, Student and F references are accurate to 0.05
  percentage points for up to 10,000 degrees of freedom. Beyond that range,
  reports flag the reference as outside its validated range.

## Evidence

- Mathematical results: {ref}`unbiasedness of OLS <formal-STATADIC-LINEAR-009>`,
  the {ref}`Gauss–Markov theorem <formal-STATADIC-LINEAR-010>`,
  {ref}`weighted least squares <formal-STATADIC-LINEAR-011>` and
  {ref}`consistency of HC0 <formal-STATADIC-LINEAR-012>`.
- Reference comparisons: coefficients, HC3 standard errors and intervals are
  matched against statsmodels in the {ref}`reference index <reference-api.regression>`.
- Studies: the {ref}`OLS and WLS <research-procedure-linear.ols-wls>` and
  {ref}`covariance <research-procedure-linear.covariance>` procedures in the
  study catalogue.

## See also

- [Your first regression](../start/first-regression.py) and the
  [OLS analysis tutorial](../tutorials/ols/index.md)
- [Change the covariance estimator](../guides/change-covariance.py)
- [Hypotheses and tests](hypotheses.md) and [reports](reporting.md)
