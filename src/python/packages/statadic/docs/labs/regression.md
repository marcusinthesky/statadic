# A fitted line is not the whole story

**Learning objective:** distinguish a point estimate from its uncertainty and
from the assumptions needed to interpret either.

## Predict first

Does switching from model-based covariance to HC3 change the fitted slope? Does
it remove a contaminated observation?

## Experiment

The generating mean is `1 + 2*x`, with 80 fixed-design observations. Increase
**uneven error spread**, then compare the native reports. Turn on the outcome
contamination and observe how the OLS fit responds. The underlying noise draw
stays fixed when you move controls.

The lab calls the same standalone example as the
[four-part OLS chapter](../tutorials/ols/index.md). It uses `Observations`,
`Intercept() + Linear("x")` and `Regression.ols("y", terms)`. Native fit validity
is checked before inference is presented. The diagnostics tab requests
Jarque–Bera, both Breusch–Pagan formulations, White, ordered BG/DW, both RESET
references and influence; none chooses the covariance for you.

## Read the reference convention

| Choice | Covariance | Reference |
| --- | --- | --- |
| HC3 | `HC("HC3")` | Normal, asymptotic |
| Model-based | `ModelBased()` | Student t, with the fit's residual degrees of freedom |
| HAC | `HAC(bandwidth, order_by="time")` | Normal, asymptotic; Bartlett weights |

With data fixed, these choices do not change the OLS point estimate. Their
intervals differ in covariance construction **and** reference law. HC3 is not
outlier-resistant regression and does not repair endogeneity, dependence or a
misspecified causal interpretation.

Residual patterns can suggest a problem; a plot cannot establish all assumptions.
Classical exact t inference requires stronger model conditions than an arbitrary
heteroskedastic or contaminated dataset satisfies. The notebook deliberately lets
you violate those conditions to study the consequences.

BG, DW and HAC use the declared unique numeric `time` key, ascending. Here it is
observation position: bandwidth and lags count observations, without inferring
calendar intervals or equally spaced sampling. The named-hypothesis tab tests
the coefficient of `x` against zero, using a classical F reference for model-based
covariance and chi-square references for HC3/HAC.

[Next: observations that are distributions](distributional-workflow.md)

```{raw} html
<a href="snapshots/regression.html">View the executed static snapshot</a> · <a href="downloads/regression.py">Download the notebook</a> · <a href="snapshots/regression.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
