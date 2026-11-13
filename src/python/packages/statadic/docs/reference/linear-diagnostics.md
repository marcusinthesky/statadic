---
type: reference
description: Test the assumptions behind an ordinary least squares fit and measure how much each row drives its estimates.
---

# Linear diagnostics

## Usage

```python
import random

from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import (
    RESET,
    BreuschGodfrey,
    BreuschPagan,
    DurbinWatson,
    Influence,
    JarqueBera,
    White,
)
from statadic.api.inference import HC
from statadic.api.regression import Regression

rng = random.Random(7)
x = [i / 10 for i in range(60)]
data = {"time": list(range(60)), "x": x, "y": [0.3 + 0.7 * v + rng.gauss(0, 1) for v in x]}

fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
diagnostics = fit.diagnose(
    JarqueBera(),
    BreuschPagan(),
    White(),
    RESET(),
    BreuschGodfrey(lags=2, order_by="time"),
    DurbinWatson(order_by="time"),
    Influence(),
)
print(diagnostics.report().to_text())
print(diagnostics["breusch_pagan"].pvalue)  # one outcome, by name
uncertainty = fit.infer(covariance=HC("HC3"))  # a separate, explicit choice
```

## Description

`result.diagnose(*requests)` evaluates each requested procedure against the
residuals of an ordinary least squares fit. Estimation, diagnosis and inference
stay separate, so a diagnostic never refits the model, changes its estimates or
chooses a covariance on your behalf.

Every request names one procedure, with its options and its null hypothesis.
There is no request that runs everything, and each result family accepts only the
requests it supports. Outcomes keep the order of the request and carry unique
names, so pass `name=` to ask for the same procedure twice. Tests return a
`TestOutcome` holding a structured [Hypothesis](hypotheses.md), a reference
distribution, its assumptions and a record of how it was built, while influence
returns descriptive quantities instead of a test.

The JAX kernels live in `statadic.inference.linear_diagnostics`, where they take
a `LinearFit` and support `jit` and `vmap`.

## Options

Each test states a null hypothesis and compares its statistic with a reference
distribution, and the statistics themselves are defined under Methods and formulas.

| Request | Null hypothesis | Reference |
| --- | --- | --- |
| `JarqueBera()` | Errors have skewness 0 and kurtosis 3 | chi-square(2) {cite:p}`jarque-bera-1987` |
| `BreuschPagan()` | Error variance does not depend on the mean design's varying columns | chi-square(rank - 1) {cite:p}`breusch-pagan-1979,koenker-1981` |
| `BreuschPagan(variables=("z",))` | Error variance does not depend on the named training columns | chi-square(rank - 1) {cite:p}`koenker-1981` |
| `BreuschPagan(studentized=False)` | As above, with Gaussian errors | chi-square(rank - 1) {cite:p}`breusch-pagan-1979` |
| `White()` | Error variance does not depend on regressors, squares or cross-products | chi-square(rank - 1) {cite:p}`white-1980` |
| `DurbinWatson(order_by="time")` | No first-order autocorrelation in the declared order | None, with a statistic in $[0, 4]$ {cite:p}`durbin-watson-1950` |
| `BreuschGodfrey(lags=L, order_by="time")` | No autocorrelation at lags 1 to $L$ in the declared order | chi-square($L$) {cite:p}`breusch-1978,godfrey-1978` |
| `RESET(powers=(2, 3), reference="chi2")` | Added powers of the fitted values have zero coefficients | chi-square($q$), or $F(q, n - k)$ with `reference="F"` {cite:p}`ramsey-1969` |
| `Influence()` | None, since it describes sensitivity | None {cite:p}`cook-1977,belsley-kuh-welsch-1980` |

`DurbinWatson` and `BreuschGodfrey` require `order_by`, either a column name or
`None` for the current row order, following the rule described for `HAC` in the
[regression reference](regression.md). Lags count adjacent observations in that
order.

## Results

`fit.diagnose(...)` returns a `Diagnostics` collection that you can index by
name, iterate over in request order or summarise with `report()` as a combined
[report](reporting.md). Unavailable outcomes stay in the collection and appear in
the report with a reason. A `TestOutcome` exposes `statistic`, `pvalue`,
`reference`, `available`, `assumptions` and `sources` for further use.

A residual test needs a valid, identified fit with positive residual degrees of
freedom and a non-zero residual sum of squares. When one of these conditions
fails, the outcome reports one of these reasons:

| Reason | Meaning |
| --- | --- |
| `invalid-fit` | The point fit is unavailable |
| `weighted-fit` | Relative precisions differ, so the fit is WLS |
| `no-residual-degrees-of-freedom` | The fit is saturated |
| `variance-design-lacks-constant-or-variation` | Breusch-Pagan needs a constant and one varying regressor |
| `design-lacks-constant-or-variation` | White and Breusch-Godfrey need a constant in the mean design |
| `auxiliary-lag-design-unavailable` | The residual-lag design is unavailable |
| `added-powers-not-identified` | The fitted-value powers add no identified column |
| `linear-influence-ineligible` | The full fit is ineligible, so check each mask below |

Influence holds one array for each of `leverage`, `studentized_internal`,
`studentized_external`, `cooks_distance`, `dffits` and `dfbetas`. Each has a
validity mask named like `dffits_valid`, with one entry per row or, for
`dfbetas`, per row and coefficient, and unavailable entries are `NaN`. A deleted
fit with zero variance makes external studentization, DFFITS and DFBETAS
unavailable for that row alone.

## Methods and formulas

Let $e$ be the residual vector, $n$ the number of rows and $R^2$ the fit of an
auxiliary regression. Jarque-Bera uses $\tfrac{n}{6}\left(S^2 +
\tfrac{(K-3)^2}{4}\right)$ with skewness $S$ and kurtosis $K$ from the biased
residual moments. The studentized Breusch-Pagan statistic is $nR^2$ from
regressing $e^2$ on the variance design, and the original form instead halves the
explained sum of squares of $e^2 / \operatorname{mean}(e^2)$. White applies the
studentized form to the regressors together with all of their products.

Durbin-Watson is the ratio $\sum_t (e_t - e_{t-1})^2 / \sum_t e_t^2$, and
Breusch-Godfrey is $nR^2$ from regressing $e$ on the design and $L$ zero-filled
lags, so that all $n$ rows enter. RESET is the classical Wald statistic of the
fit augmented with powers of the fitted values. White and RESET count only
identified added columns, which can lower their degrees of freedom.

Influence reports leverage $h_i$, internal and external studentized residuals,
Cook's distance, DFFITS and DFBETAS from deleting row $i$.

## Assumptions and limits

Each test is valid only under its own conditions, which this page states once
for every procedure.

- Jarque-Bera needs independent, identically distributed Gaussian errors, and
  its chi-square reference is asymptotic.
- Breusch-Pagan and White need a correct conditional mean, independent rows and
  score-test asymptotics, and the original Breusch-Pagan form also needs
  Gaussian errors.
- Durbin-Watson has a statistic but no p-value, since its null distribution
  depends on the design.
- Breusch-Godfrey needs a correct mean and asymptotics for the declared order.
  Sorting does not establish equal spacing or stationarity, and trimming the
  first $L$ rows would define a different procedure.
- RESET needs a correct mean under the null and homoskedastic, independent
  errors, with a reference resting on the classical F or an asymptotic
  chi-square.
- Influence marks sensitivity to one row, so a large Cook's distance does not
  make an observation wrong.

Reading the results needs the same care.

- Failing to reject is not evidence that an assumption holds.
- Normality is not a Gauss-Markov condition, since OLS is BLUE without it.
- Robust covariance such as `HC("HC3")` changes inference but does not test for
  heteroskedasticity.
- Residuals are not errors, so exogeneity cannot be checked from them.
- Weighted fits are unavailable, and Ljung-Box, Goldfeld-Quandt and the
  D'Agostino omnibus test are not implemented.
- Panel, GLM and spatial results raise a `TypeError` for these requests, and
  spatial results have their own [diagnostics](spatial-panels.md).

## Evidence

- Mathematical results: the {ref}`Durbin-Watson range <formal-STATADIC-DW-001>`,
  the {ref}`Jarque-Bera zero set <formal-STATADIC-JB-001>` and the
  {ref}`deleted-row studentization identity <formal-STATADIC-INFLUENCE-001>`.
- Reference comparisons: statistics and p-values are matched against statsmodels
  in the {ref}`reference index <reference-inference.linear_diagnostics>`, and
  influence is matched against explicit leave-one-out refits.
- Studies: the Gaussian-null size study in the
  {ref}`residual diagnostics procedure <research-procedure-linear.residual-diagnostics>`.

## See also

- [Diagnose an OLS fit](../tutorials/ols/diagnose.py) and the reference for
  [hypotheses and tests](hypotheses.md)
- [Regression](regression.md) and [reports](reporting.md) for the surrounding workflow
