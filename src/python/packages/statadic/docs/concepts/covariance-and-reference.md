---
type: explanation
description: How covariance estimates and reference distributions turn a point estimate into an interval or test.
---

# Covariance and reference distributions

Two analysts can estimate the same regression coefficients yet report different
standard errors because their calculations rely on different assumptions about
how the observations vary together. That distinction becomes especially useful
when the data include repeated observations from the same group.

## Estimate the model, then request uncertainty

The fitted result keeps the coefficients and the design built from the training
rows. Calling `infer` requests a covariance estimator for those coefficients.
Changing that request leaves the fitted coefficients unchanged.

Classical covariance describes the model's error assumptions. HC estimators
allow heteroskedasticity, clustered covariance allows dependence within named
groups, and HAC covariance describes dependence across ordered observations.
Choose among them using the sampling design and the method's stated scope.

## Choose a reference distribution

A covariance estimate provides a scale for uncertainty, while a reference
distribution turns a statistic on that scale into an interval or test. A normal
reference and a Student reference therefore answer a further question beyond
the choice of covariance, with degrees of freedom and finite-sample corrections
remaining explicit.

Combining robust covariance with a Student reference gives an approximation
whose interpretation depends on the assumptions documented for that combination.
Small samples need particular care. Choosing a reference distribution alone
cannot correct an unsuitable covariance model or establish interval coverage.

## Keep the normalization explicit

For estimating equations, the variability of a sample mean differs from that
of its square-root-sample-size scaling. `sandwich_covariance` returns the
limiting covariance for the latter convention, while `parameter_covariance`
divides by the observation count and applies its explicit correction.

Use [Regression](../reference/regression.md) for supported combinations and
[resampling](../reference/resampling.md) when uncertainty is calculated from
repeated draws. The [evidence pages](../validation/index.md) describe the
configurations checked for numerical agreement and statistical coverage.
