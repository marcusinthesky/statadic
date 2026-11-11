# 1. Estimate and inspect the line

The generating mean is $1+2x$. A local random generator supplies Gaussian noise;
the lab can increase its spread with $|x|$ or contaminate the final outcome.
The defaults use seed 17, constant error spread and no contamination. The `time`
column is a unique observation-position key for later ordered calculations.

```{literalinclude} ../../../examples/ols_analysis.py
:start-after: "# ols-fit-start"
:end-before: "# ols-fit-end"
:language: python
:dedent: 4
```

`Regression.ols` fixes the ordinary least-squares composition. The terms include
an intercept explicitly. `estimate` retains the actual design and numerical
fit; `require_valid` stops this example if that fit is unavailable. It does not
check exogeneity, independence or a sampling model.

<!-- statadic:ols-result fit -->

The rank counts identified design columns; residual degrees of freedom are
$n-\operatorname{rank}(X)$. SSE is $\sum_i(y_i-\hat y_i)^2$. This example also
derives centered $R^2=1-\mathrm{SSE}/\sum_i(y_i-\bar y)^2$. That interpretation
uses this design's intercept and a nonconstant outcome. A high $R^2$ alone does
not support a causal interpretation or guarantee useful new-row predictions.

<details>
<summary>Reproduce the data</summary>

```{literalinclude} ../../../examples/ols_analysis.py
:start-after: "# ols-data-start"
:end-before: "# ols-data-end"
:language: python
```

</details>

**Mathematical result:** {ref}`OLS unbiasedness <formal-STATADIC-LINEAR-009>` and
{ref}`Gauss–Markov <formal-STATADIC-LINEAR-010>` state their design, mean and variance
assumptions; they do not verify this floating-point fit.
**Numerical comparison:** {ref}`the constructor reference index <reference-api.regression>`
records matched configurations for coefficients, HC3 standard errors and
intervals. Those reference fixtures differ from this chapter's sample.
**Study design:** {ref}`OLS/WLS studies <research-procedure-linear.ols-wls>`
list their own designs and dispositions; registration is not current-release
qualification.

Continue with [residual diagnostics](diagnose.md).
