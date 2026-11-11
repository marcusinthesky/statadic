# 3. Choose covariance and reference separately

A covariance estimate and a reference distribution answer different questions.
The same fitted coefficients below receive three explicit uncertainty
calculations. None is chosen by the diagnostic p-values.

```{literalinclude} ../../../examples/ols_analysis.py
:start-after: "# ols-infer-start"
:end-before: "# ols-infer-end"
:language: python
:dedent: 4
```

<!-- statadic:ols-result infer -->

Model-based covariance uses the estimated homoskedastic error variance. Student
t intervals and an F test with residual degrees of freedom have their classical
finite-sample justification under independent Gaussian errors and the correctly
specified full-rank linear model.

HC3 adjusts squared residual contributions for leverage. Here it is paired with
normal intervals and a chi-square restriction test: an asymptotic procedure,
not an exact small-sample replacement. It neither changes coefficients nor makes
OLS resistant to contaminated outcomes.

Bartlett HAC includes serially lagged score products through the specified
bandwidth. Here the bandwidth is three observations in ascending `time`, with
the default correction factor of one. Its normal/chi-square references need
appropriate dependence and bandwidth asymptotics. Sorting alone supplies none
of these conditions.

The named `Hypothesis(LinearRestriction.zero("x"))` tests the fitted coefficient
of `x`, not the generating value 2. Its `TestOutcome` retains the hypothesis,
statistic, p-value when calibrated, assumptions and construction. See
[hypotheses](../../reference/hypotheses.md) for multi-row restrictions. The report
keeps unavailable intervals visible rather than replacing them with another law.

**Mathematical result:** {ref}`HC3 adjustment bounds <formal-STATADIC-HC-003>` are
distinct from {ref}`HC3 consistency assumptions <formal-STATADIC-HC-004>` and from
interval coverage. **Numerical comparison:**
{ref}`HC3 reference controls <reference-linear.covariance.hc3>` and the exact
HAC rows in the [reference index](../../validation/references.md) state matched
covariance, correction and reference conventions. **Study design:**
{ref}`covariance studies <research-procedure-linear.covariance>` describe their
own size, power or coverage targets, not universal validity.

Continue with [prediction and retained reports](report.md).
