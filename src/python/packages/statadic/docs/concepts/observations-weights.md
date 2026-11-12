---
type: explanation
description: How observation identities and the meanings of weights determine an analysis.
---

# Observations and weights

In a scalar regression, each row usually records one observation of the
response and its predictors. When a row instead contains an empirical
distribution, its atoms describe variation within that observation rather than
additional independent units in the sample.

## Identity before shape

Matching array lengths is only a first check. Two columns can have the same
length while describing different observations, which is why `Observations`
aligns numeric, categorical and measure columns through their row identities.
Selection applies to those roles together, and a `MeasureColumn` follows the
same outer identities in the same order.

When predicting for new observations, the model applies the design learned
from its training data rather than learning another set of transformations.
Observation identities label the rows, while parameter names label the design
columns, so preserve both when preparing data for an analysis.

## Four meanings of weight

| Weight | Meaning |
| --- | --- |
| Atom mass | Probability attached to an atom within one empirical distribution |
| Relative precision | Inverse relative conditional error variance in WLS |
| Graph weight | Strength of a spatial connection under the chosen normalization |
| Mixture weight | Contribution of a whole distribution to a mixture |

When atom masses are omitted, each atom in the empirical distribution receives
the same mass, whereas supplied masses follow the normalization contract of
their carrier. Padding contributes zero mass.

For WLS, the model estimates a common variance scale from the observations and
their precision-weighted residuals. Multiplying all relative precisions by one
positive constant leaves the fitted coefficients and estimated-scale covariance
unchanged. The weights describe precision rather than extra observations.

## Choose the sampling unit

Before estimating a model, state what a row represents and which units can be
treated as independent. For dyadic observations, shared endpoints can create
dependence even when each pair occupies a different row.

Use [Regression](../reference/regression.md) for the input contract and
[covariance and references](covariance-and-reference.md) for uncertainty that
matches the sampling design.
