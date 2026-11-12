---
type: reference
description: Fit parametric accelerated failure time models to right-censored durations, with repeated spells, spatial dependence and predictions on several scales.
---

# Survival

## Usage

```python
import jax.numpy as jnp

from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import ModelBased
from statadic.api.regression import Regression
from statadic.api.survival import RightCensored
from statadic.model.survival import LogNormalAFT

spells = Observations(
    {
        "x": jnp.array([-1.0, -0.7, -0.2, 0.1, 0.3, 0.6, 0.9, 1.2]),
        "duration": jnp.array([0.8, 1.6, 0.9, 1.7, 2.4, 1.3, 2.1, 3.0]),
        "event": jnp.array([1, 1, 1, 0, 1, 1, 0, 1]),
        "phase": jnp.array([0.05, 0.2, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95]),
    }
)
model = Regression(
    RightCensored("duration", "event"),
    Intercept() + Linear("x"),
    law=LogNormalAFT(),
)
result = model.estimate(spells).require_valid()
medians = result.predict(spells)
curves = result.predict(spells, jnp.array([0.5, 1.0, 2.0]), kind="survival", grid=True)
inference = result.infer(covariance=ModelBased())
```

## Description

Passing a {py:class}`~statadic.api.survival.RightCensored` response to
`Regression` fits elapsed durations with an accelerated failure time (AFT)
law, in which the logarithm of the duration is linear in the predictors plus a
scaled error. Each row is one spell, where `event=1` marks an observed event
and `event=0` marks right censoring, and durations must be positive and finite.
Optional delayed entry conditions the likelihood on survival to the entry time,
with `0 <= entry < duration`.

Every fit needs an explicit `law`. Each prediction is requested from the
fitted result and names the quantity it returns, so one fit can yield medians,
survival curves or restricted means without refitting.

## Options

| Law | Ancillary parameter | Notes |
| --- | --- | --- |
| `WeibullAFT()` | Log shape | Monotone hazard |
| `LogNormalAFT()` | Log scale | Only law for spatial survival |
| `LogLogisticAFT()` | Log shape | Finite mean needs shape above one |

Delayed entry uses `RightCensored("duration", "event", entry="entry")`, and a
substituted estimator must match the chosen law. Repeated spells add a panel
structure and fixed effects:

```python
from statadic.api.data import PanelLayout
from statadic.model.survival import WeibullAFT

panel_spells = Observations(
    {
        "x": [-1.0, -0.7, -0.2, 0.1, 0.3, 0.6, 0.9, 1.2],
        "duration": [0.8, 1.6, 0.9, 1.7, 2.4, 1.3, 2.1, 3.0],
        "event": [1, 1, 1, 0, 1, 1, 0, 1],
        "unit": [0, 1, 2, 3, 0, 1, 2, 3],
        "period": [0, 0, 0, 0, 1, 1, 1, 1],
    },
    row_ids=("a0", "b0", "c0", "d0", "a1", "b1", "c1", "d1"),
)
panel_model = Regression(
    RightCensored("duration", "event"),
    Linear("x"),
    law=WeibullAFT(),
    structure=PanelLayout("unit", "period"),
    effects="time",
)
panel_result = panel_model.estimate(panel_spells)
```

Effects can be `"pooled"`, `"unit"`, `"time"` or `"two-way"`, and each
entity-period holds at most one spell in a panel that may be unbalanced. Unit,
time and two-way effects supply their own location, so an explicit `Intercept()`
in the terms is rejected.

Spatial dependence in log duration uses `SpatialLag("graph")` with a
`LogNormalAFT()` law and the `SpatialSurvivalML` estimator. Its inputs are
`PanelData`, which keeps covariates for every active graph node apart from the
observed spells, and `SpatialSurvivalData`, which adds the integration draws.
Missing outcomes are integrated out of the likelihood, and the
[spatial panel survival tutorial](../tutorials/spatial-panel-survival.py)
walks through a complete fit with `refine(prepared, ...)`.

## Results

`predict(data, value, kind=..., grid=..., entry=...)` accepts `Observations`,
named mappings and eager Arrow-compatible frames, which are adapted to the
training predictor roles. Mappings broadcast scalars, and fixed effects
need the trained entity and period keys.

| `kind` | `value` | Result |
| --- | --- | --- |
| `"median"` (default) | None | Marginal median duration |
| `"quantile"` | Probability | Duration quantile |
| `"survival"` | Time | Survival probability, optionally conditional on `entry` |
| `"cumulative_hazard"` | Time | Cumulative hazard, optionally conditional on `entry` |
| `"restricted_mean"` | Horizon | Mean duration up to the horizon |
| `"mean"` | None | Unconditional mean duration |

A scalar `value` broadcasts over observations, and a vector is row-aligned.
With `grid=True`, a one-dimensional `value` yields an array with one row per
observation and one column per value. The median and the mean take only the
data. A log-logistic fit with shape at or below one has an infinite mean, so
its mean prediction is NaN while its other predictions stay available. Its
restricted mean uses bounded quadrature and returns NaN for unresolved integrals.

`result.infer(covariance=ModelBased())` uses the covariance from the observed
information and records whether it is available. A structural contrast compares
two predictor settings on the scale of the linear predictor, using the basis
learned from the training rows:

```python
from statadic.api.design import CyclicSpline

season = CyclicSpline("phase", df=3, period=1.0)
seasonal = Regression(
    RightCensored("duration", "event"),
    Intercept() + Linear("x") + season,
    law=LogNormalAFT(),
)
fit = seasonal.estimate(spells).require_valid()
contrast = fit.design_contrast(season, at=0.25, versus=0.75)
test = fit.infer(covariance=ModelBased()).contrast(contrast, reference="normal")
print(test.estimate, test.standard_error, test.valid)
```

A spatial fit offers the same `design_contrast` for a structural contrast of
log duration, and its predictions are unconditional marginals for each node,
grouped by period and ordered by the identity of the graph.

## Methods and formulas

For a spell with duration $t_i$, entry $a_i$ and design row $x_i$, the model
is $\log T_i = x_i^\top\beta + \sigma\varepsilon_i$ with $\varepsilon_i$
following the extreme-value, normal or logistic law. An event contributes the
density $f(t_i)$ and a censored spell contributes the survival function
$S(t_i)$, while delayed entry divides each contribution by $S(a_i)$. Maximum
likelihood maximises the sum of the logarithms of these contributions.

## Assumptions and limits

- Censoring and entry must be independent of the duration given the
  predictors, and numerical availability gives no test of this assumption.
- Spells are elapsed times, which differs from a start-stop interface in the
  style of counting processes.
- Fixed effects add neither frailty nor a model for dependence between the
  repeated spells of one entity, and new levels of an effect are rejected.
- Two-way coding uses one time reference, so a disconnected incidence of units
  and periods leaves dependent columns and is rejected as rank deficient.
- Model-based inference is conditional on the model and gives no promise of
  calibration in finite samples.
- Spatial survival needs a dense graph likelihood, supports float32 and
  float64, and rejects positive entry. Small integration budgets can mislead
  for heavily censored outcomes, so inspect the diagnostics of each fit.
- Preparation, reports and refinement run eagerly, outside of `jax.jit`.

## Evidence

- Mathematical results: the {ref}`delayed-entry likelihood <formal-STATADIC-SURVIVAL-001>`,
  the {ref}`AFT transformation <formal-STATADIC-SURVIVAL-002>` and
  {ref}`independent censoring <formal-STATADIC-SURVIVAL-003>`, plus the
  {ref}`Weibull <formal-STATADIC-SURVIVAL-004>` and
  {ref}`lognormal <formal-STATADIC-SURVIVAL-005>` densities.
- Reference comparisons: Weibull and lognormal fits, covariances and
  predictions are matched against lifelines, and spatial survival against
  spreg reductions.
- Studies: the {ref}`independent <research-procedure-survival.independent>`,
  {ref}`spatial <research-procedure-survival.spatial>` and
  {ref}`mean <research-procedure-survival.mean>` procedures in the study
  catalogue. Some cells are inconclusive, so read the known limitations before
  relying on repeated-spell or spatial Wald inference.

## See also

- [Choose a survival model](../concepts/choosing-a-survival-model.md)
- [Spatial panel survival tutorial](../tutorials/spatial-panel-survival.py)
- [Regression](regression.md) and [response laws](conditional-laws.md)
