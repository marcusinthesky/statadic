---
type: reference
description: Find covariance estimators and numerical operations for sampling uncertainty.
---

# Covariance and error

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
    {"x": [0., 1., 2., 3.], "y": [1., 2., 2.8, 4.2]}
).require_valid()
from statadic.api.inference import HC

inference = fit.infer(covariance=HC("HC3"))
```

## Description

A covariance estimator describes uncertainty in fitted parameters. Other covariance operations describe observations or construct numerical covariance matrices, so choose the object that matches your question.

For a regression analysis, the choice of covariance reflects the dependence and variation that you expect among the observations.

## Options

| you want | call | notes |
|---|---|---|
| classical, HC, HAC or one-way cluster covariance | `statadic.inference.linear_covariance` | preview. Actual fit owns row/weight/parameter alignment. Correction and reference choices remain separate |
| actual-weight GMM sampling covariance | `statadic.inference.gmm.covariance.gmm_covariance` | full sandwich from stored weight and final moment variability, not an efficient-weight shortcut |
| model-information or HC covariance for a GLM fit | `statadic.inference.glm_covariance.glm_covariance` | shared mean-equation composer. Fixed dispersion one and explicit correction |
| checked linear restrictions and coefficient intervals | `statadic.inference.parameters` | preview. Explicit chi-square/F or normal/Student reference, separate degrees of freedom and availability status |
| a HAC / long-run covariance | `statadic.operators.longrun` | `newey_west`, `hall_centered_hac`, `bartlett_weights`, `optimal_bandwidth` |
| limiting covariance from a mean estimating equation | `statadic.core.equations.sandwich_covariance` | accepts a derivative panel or `MeanEquationJacobian`. No division by observation count |
| observation-level parameter sampling covariance | `statadic.core.equations.parameter_covariance` | explicit stored observation count and separate correction. Inspect identification/numerical flags. Component alignment remains the caller's obligation |
| a covariance operator from distances and marginal variances | `statadic.operators.covariance` | `sigma_dist` and heterogeneous-κ `sigma_dist_hetero_kappa` |
| sample covariance or linear shrinkage toward a target | `statadic.estimation.covariance` | fixed/independent targets use guarded `optimal_lambda_dist`. Same-sample targets use the explicitly heuristic `plug_in_lambda_dist` |
| a diagonal-preserving PSD repair | `statadic.optimize.psd` | Qi–Sun semismooth Newton in correlation space. Ridge, clipping, diagonal-shrink and array-valued distortion diagnostics are also available |
| a rank-clean undirected dyadic design | `statadic.operators.design` | symmetric endpoint effects (`node_effect` by default, caller-configurable label prefix) and Frisch--Waugh--Lovell residualization |
| both FWL residual series and their slope | `statadic.operators.design.fwl_residualize` | array-valued `FWLResiduals` pytree. Fixed-shape JIT/batching with a static focal index. Caller precision and rank-clean design. `fwl_coefficient` returns an eager Python float |
| multinomial node-bootstrap inference for a dyadic regression | `statadic.inference.dyadic` | Consumes model specifications from `statadic.model.dyadic`. Point fits live in `statadic.estimation.dyadic`. Dataframe schemas, labels and publication ladders stay with callers |

## Results

Inference retains its fit and the covariance requested by the caller. Inspect availability before interpreting intervals or tests.

## Assumptions and limits

A covariance estimate relies on its sampling assumptions. Corrections and reference distributions are separate choices, and a numerical result does not establish those assumptions.

## Evidence

[Independent comparisons and studies](../validation/index.md) describe the scope of evidence for each method.

## See also

- [Regression](regression.md)
- [Resampling](resampling.md)
