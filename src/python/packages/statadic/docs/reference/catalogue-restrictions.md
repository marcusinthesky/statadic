---
type: reference
description: Find tests for moment equalities, inequalities and boundary hypotheses.
---

# Moment restrictions

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
    {"x": [0., 1., 2., 3.], "y": [1., 2., 2.8, 4.2]}
).require_valid()
```

## Description

Moment restrictions express claims about a model that can be tested against
observed data. The procedures below differ in their requirements for covariance,
calibration and identification.

The hypothesis and its calibration have separate roles, so inspect both when interpreting evidence from an estimated model.

A restriction can concern an equality or an inequality, and the appropriate reference distribution depends on the question and the estimator.
For a test of a fitted model, check the requirements for its reference
distribution alongside the assumptions of the estimator.

## Options

| you want | call | notes |
|---|---|---|
| a fixed-moment HAC Wald test | `statadic.inference.gmm.gmm_wald` | fails closed if the HAC moment covariance loses rank |
| conventional J inference from a stored affine fit | `statadic.inference.gmm.hansen_j_from_fit` | requires explicit correct-specification acknowledgement and efficient-weight evidence. Unavailable for exactly identified or arbitrary fixed-weight fits |
| named host restrictions or stored-fit diagnostics | `statadic.api.hypotheses.Hypothesis`, `statadic.api.diagnostics.HansenJ` | Hypotheses state claims. Procedures retain their own reference, assumptions and numerical availability. See [hypotheses](hypotheses.md). |
| fixed or affine generalized empirical likelihood fitting | `statadic.estimation.gel.FixedGEL` / `AffineGEL` | [GEL workflow](moment-estimators.md). Retains multipliers, structural parameters and separate solver/rank evidence |
| GEL test calibration from the actual fit | `statadic.inference.gel.gel_test` | Fixed fits support LR/LM/Wald. Affine fits support LR. Chi-square inference requires explicit iid/serially-uncorrelated moments |
| moment inequalities | `statadic.inference.wolak.wolak_test`, `statadic.inference.gms.indicator_gms` | Wolak uses a χ-bar-square reference. Indicator GMS is explicitly uncorrected unless a caller supplies a correction, which is not certified as Table-I calibration |
| reduce a large pair-moment panel | `statadic.inference.clustering.cluster_pairs` | drops empty clusters so they cannot fabricate rank or degrees of freedom |
| κ uncertainty and a lower-boundary test | `statadic.inference.kappa` | delta method, Jacobian-concentration diagnostic, and the `½χ²₀+½χ²₁` boundary tail |

## Results

Each test returns a statistic, its reference distribution and numerical
availability. A host hypothesis states the claim, while a diagnostic or test
specifies the procedure used to evaluate it.

## Assumptions and limits

Hansen J inference depends on the efficient-weight and specification conditions
described for the fit. Generalized empirical likelihood and inequality tests
have separate assumptions for their calibration.

## Evidence

The [independent comparisons and studies](../validation/index.md) describe the
configurations for which evidence is available for each method.

## See also

- [Moment estimators](moment-estimators.md)
- [Hypotheses](hypotheses.md)
