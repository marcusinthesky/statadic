# Compose survival, panel identity and spatial dependence

A familiar progression across survival, panel and spatial tools is to ask first
how duration varies with predictors, then how outcomes differ across periods,
and finally whether neighboring units' log durations are dependent. This
example makes those choices explicit with STATADIC components and one small,
synthetic dataset. It does not establish causality or qualify a spatial model
for production use.

The questions overlap the panel workflows familiar from
[linearmodels](https://bashtage.github.io/linearmodels/panel/index.html), spatial
regression in [spreg](https://pysal.org/spreg/), and survival analysis in
[lifelines](https://lifelines.readthedocs.io/en/latest/). Here they meet in one
model with explicit component contracts.

## Build the question in stages

Each of six units has one positive elapsed-time spell in each of three periods.
The first unit in the first period has an administrative observation limit of
1.0, specified before observing its duration; the other limits are infinite.
Its latent duration exceeds the limit, so one of the 18 spells is right-censored
at time 1.0 and has `event=0`. The other 17 events are observed. The starting model uses a lognormal
accelerated failure-time distribution and an intercept plus `x`. A second fit
adds period effects through `PanelLayout` and `effects="time"`. The independent
spell likelihood does not add frailty or dependence between repeated spells.
The fixed-effect models omit `Intercept()`; their effects supply the location.

For the spatial version, let `T_t` be the vector of durations, `W_t` the
explicit graph, `X_t` the design and `a_t` the time effect in period `t`. The
spatial log-time specification has the form

$$
\log T_t = \rho W_t \log T_t + X_t\beta + a_t + \varepsilon_t.
$$

The fixed value of `rho` describes the assumed dependence, not an estimate.
Each graph names the units in its own order. The example uses identical,
row-normalized complete graphs for the three periods, and compares a fixed
`rho=0` control with fixed `rho=0.2`. Zero is the independent-error baseline;
the nonzero fit is a sensitivity illustration, not evidence that 0.2 is the
true dependence. The sample was generated without spatial dependence.

## Run the complete example

:::{literalinclude} ../../examples/spatial_panel_survival.py
:language: python
:::

{download}`Download the script <../../examples/spatial_panel_survival.py>`.
From the repository root, with [STATADIC installed](../start/installation.md), run:

```bash
python src/python/packages/statadic/examples/spatial_panel_survival.py
```

Set `JAX_ENABLE_X64=0` or `1` before Python starts to select precision. Both precision settings were exercised on this fixture.
The script prints fit reports, model-based inference reports, and survival at
elapsed time one, ordered by graph node within the first period.

| Question | Component | What it contributes |
|---|---|---|
| What happened to each spell? | `Observations`, `RightCensored` | Aligned data and event/censoring roles |
| Which predictors enter? | `Linear("x")`, with `Intercept()` for the pooled model | Reusable named design |
| How are repeated units organized? | `PanelLayout`, `effects="time"` | Unit/period identity and reference-coded period effects |
| Which neighbors interact? | `prepare_spatial_weights` | Explicit unit order and period-specific graph |
| What is observed on the graph? | `PanelData` | Separate full-node covariates and observed spells |
| How is the joint likelihood evaluated? | `SpatialSurvivalData`, `IntegrationDesign`, `SpatialSurvivalML` | Replayable keyed draws on the input and numerical controls on the estimator |
| Is uncertainty available? | `ModelBased`, `infer` | Model-based covariance conditional on the fitted model |

## Add a cyclic seasonal effect

Suppose each district-year node also has a `phase` in the annual cycle.
Place it on the full node table so neighbors with missing outcomes still have
a seasonal predictor. Here `graphs` is an increasing sequence of
`(year, SpatialWeights)` pairs. Supply one replayable integration design per
year, in the same order.

```python
from statadic.api.design import CyclicSpline, Linear
from statadic.api.spatial.graphs import PanelData, SpatialLag
from statadic.api.inference import ModelBased
from statadic.api.data import PanelLayout
from statadic.api.survival import RightCensored, SpatialSurvivalData
from statadic.api.regression import Regression
from statadic.model.survival import LogNormalAFT

layout = PanelLayout("district", "year")
data = PanelData(
    nodes=nodes, outcomes=spells, panel=layout, graphs={"neighbors": graphs}
)
model = Regression(
    response=RightCensored("duration", "event"),
    law=LogNormalAFT(),
    terms=Linear("exposure") + CyclicSpline("phase", df=6, period=1.0),
    structure=layout,
    effects="two-way",
    dependence=SpatialLag("neighbors"),
)
fit = model.estimate(SpatialSurvivalData(data, streams)).require_valid()
median_by_node = fit.predict(data)
uncertainty = fit.infer(covariance=ModelBased())
```

The six seasonal coefficients use seven equally spaced cyclic cubic basis
functions with one reference coefficient fixed at zero. The fixed effects supply
the location component. Values wrap at each integer year with matching first and
second derivatives. Normalize dates to phase using a documented leap-day
convention. A phase constant within every year is absorbed by the year effects
and the full-node rank check will reject that design.

This specification estimates a structural effect on log duration. Predicted
marginal medians also reflect propagation through the fitted spatial graph.
The result retains the graph-ordered design, fixed-effect coding and the actual
integration inputs. Its report records period keys, draw budgets, convergence
and information availability. The spatial survival backend supports a
lognormal outcome lag with zero entry. The spline uses a fixed basis with no
smoothing-parameter estimation.

## Read the reports

Check fit validity, identification, convergence and integration status before
interpreting parameters. The ordinary fit report has no standard errors until
inference is requested. In this fixture, model-based covariance was available
for both spatial fits in both precision modes. The inference report leaves test
statistics, p-values and intervals unavailable when no reference distribution
is selected; that is an explicit choice, not evidence of a failed fit. A valid
fit or covariance does not guarantee correct model assumptions, calibrated
inference, a global optimum, or accurate numerical integration.

The two-draw-replicate, 16-draw designs keep the lesson small. One node has a censored spell; this small example does not establish the
accuracy of integration for censored or missing nodes. The full-node and observed-spell
inputs remain separate because missing graph outcomes are integrated out, not
replaced with zero durations. Predictions are unconditional marginal node
survival probabilities, not conditional on observed neighbors' durations.
Spatial positive delayed entry is unsupported. For diagnostics, refinement and
scope, read [the survival guide](../survival-models.md).
