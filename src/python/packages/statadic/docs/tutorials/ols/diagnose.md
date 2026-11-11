# 2. Ask specific questions of the residuals

Request diagnostics explicitly from the fitted result. Each outcome retains its
own null hypothesis, reference and availability; influence is descriptive.
Names distinguish the two Breusch–Pagan and two RESET formulations.

```{literalinclude} ../../../examples/ols_analysis.py
:start-after: "# ols-diagnose-start"
:end-before: "# ols-diagnose-end"
:language: python
:dedent: 4
```

<!-- statadic:ols-result diagnose -->

Jarque–Bera examines residual skewness and kurtosis. Studentized Breusch–Pagan
uses a variance regression; its original formulation additionally invokes
Gaussian errors. White includes squares and cross-products. These tests address
particular alternatives, rather than establishing constant variance when their
p-values are large.

Breusch–Godfrey tests the requested two serial lags, zero-filling pre-sample
residuals and retaining all observations. Durbin–Watson summarizes adjacent
residual differences and supplies **no p-value**. Both use ascending `time`.
An explicit `order_by=None` would acknowledge retained row order instead.
Keys must be finite, numeric and unique; ordering establishes neither equal
spacing nor stationarity, and lags count observations rather than calendar time.

RESET adds powers of fitted values and evaluates them with the requested
chi-square or classical F reference. Influence retains row-aligned leverage,
studentized residuals, Cook's distance, DFFITS and DFBETAS with separate masks.
A large value marks sensitivity; it is not permission to delete an observation.
An unavailable outcome stays visible with a reason.

No request changes the retained line or selects covariance for the next lesson.
Residuals cannot establish exogeneity, and failure to reject is not evidence for
the null. See [procedure assumptions and masks](../../linear-diagnostics.md).

**Numerical comparison:** the
{ref}`diagnostic reference index <reference-linear.residual-diagnostics.jarque-bera>`
links exact statsmodels assertions; each other procedure has its own scoped row.
**Statistical evidence:** {ref}`diagnostic study designs <research-procedure-linear.residual-diagnostics>`
and the [retained historical null-size table](../../linear-diagnostics.md#evidence)
state their sample and calibration. They do not qualify every dataset or a new
wheel. No diagnostic's source or theorem transfers to the others automatically.

Continue with [uncertainty and named hypotheses](infer.md).
