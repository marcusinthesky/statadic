---
type: reference
description: Find model estimators for linear, generalized linear, moment, panel and spatial analyses.
---

# Fitting models

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
    {"x": [0., 1., 2., 3.], "y": [1., 2., 2.8, 4.2]}
).require_valid()
```

## Description

Regression provides a common interface for models with explicit design terms. Numerical estimators are also available for callers who prepare their own problem arrays.

The reference for each family describes the preparation and assumptions that connect its numerical result to a statistical analysis.

## Options

| you want | call | notes |
|---|---|---|
| a complete-array OLS or relative-precision WLS fit | `statadic.api.regression.Regression`. Numerical component `statadic.estimation.linear.LeastSquares` over `LinearProblem` | [Linear workflow](regression.md). Explicit intercept terms, optional `precision=RelativePrecision(...)` and stored fitted state |
| predictions with checked column order | `statadic.estimation.linear.LinearFit.predict` | named fits require the same ordered column schema. Unnamed fits use positional columns |
| affine moment fitting independently of J inference | `statadic.estimation.gmm.IteratedGMM` / `FixedWeightGMM` | preview [GMM workflow](moment-estimators.md). Exactly identified fits and actual last-solve weighting are stored |
| binomial-logit or Poisson-log fitting | `statadic.api.regression.Regression` with `law=Binomial(...)` / `Poisson(...)`. Numerical components `statadic.estimation.glm.BinomialLogitMLE` / `PoissonMLE` | [GLM workflow](conditional-laws.md). Top-level predictor terms, explicit trials, offsets/exposure and finite-optimum evidence |
| GLM link, probability or expected-count prediction | `statadic.estimation.glm.GLMFit.predict` | prediction scale is explicit. Named columns must match |

| you want | call | notes |
|---|---|---|
| prepare unbalanced entity/time rows or spatial graphs | `statadic.operators.panel`, `statadic.operators.spatial` | Explicit keys, normalization, missing/singleton policy and stored order |
| pooled, fixed/random effects, between, first differences or Fama–MacBeth | `Regression(..., structure=PanelLayout(...), estimator=...)`. Numerical components `statadic.estimation.panel.Pooled` / `Within` / `Between` / `FirstDifference` / `RandomEffects` / `FamaMacBeth` over `PanelProblem` | General static panels. `statadic.inference.panel_covariance` owns uncertainty |
| SLX, SAR/SEM/SAC or selected Durbin terms | `statadic.estimation.spatial.likelihood` | Fits retain the graph, parameter schema and solver evidence |
| spatial IV or quadratic-moment GMM | `statadic.estimation.spatial.iv`, `statadic.estimation.spatial.gmm` | Explicit instruments, moments and actual last-solve state |
| balanced static spatial panels with exogenous W_t | `statadic.estimation.spatial.panel` | Actual period filters and Jacobians. KKP's distinct constant-W model lives in `statadic.estimation.spatial.kkp` |
| spatial covariance, impacts or specification diagnostics | `statadic.inference.spatial` | Uses actual fits. Reference/null and numerical availability remain separate |

## Results

A fitted result records the numerical fit and the preparation needed for prediction. Inference is requested separately.

## Assumptions and limits

A successful fit does not establish the statistical assumptions behind a model. Panel and spatial fits require the row and graph identities specified by their input contracts.

## Evidence

[Independent comparisons and studies](../validation/index.md) describe the scope of evidence for each method.

## See also

- [Regression](regression.md)
- [Moment estimators](moment-estimators.md)
- [Spatial panels](spatial-panels.md)
