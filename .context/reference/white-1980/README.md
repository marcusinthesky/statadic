---
type: article-journal
title: A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity
description: Foundational heteroskedasticity-consistent covariance estimator underlying HC0.
author: [{family: White, given: Halbert}]
issued: {date-parts: [[1980]]}
container-title: Econometrica
volume: "48"
issue: "4"
page: "817-838"
DOI: "10.2307/1912934"
URL: "https://www.jstor.org/stable/1912934"
links: []
---

# A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity

Publication metadata was checked on 2026-11-04 against the original paper's
first page, hosted by [UNLV](https://crooker.faculty.unlv.edu/econ441/econ_papers/White-Heteroskedasticity-Correction-1980.pdf),
and the DOI record. The paper supplies the heteroskedasticity-consistent
sandwich covariance construction underlying HC0. This attribution does not
establish finite-sample interval coverage or verify STATADIC's numerical implementation.

For Theorem 1(i)–(ii), the normalized score matrix replaces unobservable errors with fitted residuals. In STATADIC’s fixed-limit pathwise theorem this replacement follows by factoring the difference of squares and bounding the resulting sample average. The normalized Gram and oracle score-meat limits enter as hypotheses. The usual coefficient covariance is the normalized sandwich divided by the sample size. The package’s HC0 regression oracle compares against statsmodels on full-rank designs. [Lean source mapping](source.json) · [Reference test](../../../src/python/packages/statadic/tests/estimation/test_linear_reference.py) · [statsmodels HC0 covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.cov_HC0.html)

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
