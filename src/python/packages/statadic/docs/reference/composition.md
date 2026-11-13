---
type: reference
description: Specify a model once and reuse its preparation for estimation and prediction.
---

# Model composition

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

model = Regression.ols("y", Intercept() + Linear("x"))
data = {"x": [0., 1., 2., 3.], "y": [1., 2.1, 2.8, 4.2]}
prepared = model.prepare(data)
result = model.estimate_prepared(prepared)
predictions = result.predict({"x": [1.5, 2.5]})
```

## Description

A model specification combines a response, predictor terms and supported choices
of structure, law, dependence and estimator. Preparation aligns the observations
and learns the design state needed for estimation and prediction.

Use a named constructor for a familiar analysis, then change supported
components when the question calls for a different specification.

## Options

| Component | Reference |
| --- | --- |
| Response and design | [Regression](regression.md) |
| Learned coordinate term | [Ordination](ordination.md) |
| Response law | [Conditional laws](conditional-laws.md) |
| Instruments and moment restrictions | [Moment estimators](moment-estimators.md) |
| Panel and spatial structure | [Spatial panels](spatial-panels.md) |
| Censoring and delayed entry | [Survival](survival.md) |

## Results

`estimate(data)` prepares the data and estimates the model in one call.
`prepare(data)` followed by `estimate_prepared(prepared)` separates those steps
when an application needs to inspect or reuse preparation.

The fitted result keeps the learned transformations and parameter ordering.
Prediction applies them to new observations, and inference uses the fitted
state for the requested covariance calculation.

## Assumptions and limits

Preparation belongs to the exact model instance that created it. A different
instance, including one created with `dataclasses.replace`, rejects that state.

Supported combinations depend on the scientific family. A shared interface
cannot transfer assumptions or calibration evidence between configurations.
Learned representations must be re-estimated within a resampling procedure
when its uncertainty is intended to include representation learning.

## Evidence

The [reference comparisons](../validation/references.md) identify exact
constructions and outputs checked against independent calculations.

## See also

- [Objects in an analysis](../concepts/index.md)
- [Observations and weights](../concepts/observations-weights.md)
- [Saved artifacts](artifacts.md)
