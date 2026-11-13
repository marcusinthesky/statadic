---
type: explanation
description: How models, fitted state, inference and reports fit together.
---

# Models, fitted results, inference and reports

The coefficient printed in a table depends on which observations entered the
analysis, which model was estimated and how its uncertainty was calculated.
STATADIC keeps those decisions in separate objects so that a reader can inspect
each choice and see how it relates to the result.

## The objects in an analysis

| Object | Responsibility |
| --- | --- |
| Regression | Specifies a response, predictor terms, structure and response law |
| Ordination | Learns coordinates and the map used to place new observations |
| Estimand | Specifies a covariance or a set of moment conditions |
| Hypothesis | States the proposition evaluated by a test or diagnostic |
| Report | Presents computed values and the choices behind them |

An ordination can also serve as a regression term. The coordinates are learned
from the training predictors and the same map is applied to new rows.

## From estimation to interpretation

Calling `estimate` returns a fitted result with its numerical fit and prepared
design. Prediction uses that fitted state, while `infer` requests uncertainty
under a named covariance model. A reference distribution then supplies the
interval or test calculation.

A report presents those results together with the choices behind them,
preserving unavailable values so that a reader can distinguish a point estimate
from an interval that could be calculated for it. That distinction stays
visible when the result is exported for a manuscript.

## Explore the concepts

Read these explanations when you want to understand the choices behind a
calculation. For complete worked analyses, continue with the
[tutorials](../tutorials/index.md).

:::{toctree}
:maxdepth: 1

observations-weights
geometry
result-validity
covariance-and-reference
resampling-and-failures
choosing-a-survival-model
what-evidence-means
:::
