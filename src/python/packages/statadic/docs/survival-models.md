# Parametric survival and repeated spells

`Regression` with a `RightCensored` response fits elapsed durations with Weibull,
lognormal or log-logistic accelerated failure time (AFT) errors. A row represents one spell;
`event=1` is an observed event and `event=0` is right censoring. Durations must
be positive and finite. Optional delayed entry is conditional on survival to
entry, with `0 <= entry < duration`. These are elapsed-time spells, not a
start-stop counting-process interface.

```python
import jax.numpy as jnp

from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import ModelBased
from statadic.api.survival import RightCensored
from statadic.api.regression import Regression
from statadic.model.survival import LogNormalAFT, WeibullAFT

spells = Observations(
    {
        "x": jnp.array([-1.0, -0.7, -0.2, 0.1, 0.3, 0.6, 0.9, 1.2]),
        "duration": jnp.array([0.8, 1.6, 0.9, 1.7, 2.4, 1.3, 2.1, 3.0]),
        "event": jnp.array([1, 1, 1, 0, 1, 1, 0, 1]),
    }
)
model = Regression(
    RightCensored("duration", "event"),
    Intercept() + Linear("x"),
    law=LogNormalAFT(),
)
result = model.estimate(spells)
print(result.report())
curves = result.predict(
    spells, jnp.array([0.5, 1.0, 2.0]), kind="survival", grid=True,
)
hazards = result.predict(
    spells, jnp.array([0.5, 1.0, 2.0]), kind="cumulative_hazard", grid=True,
)
medians = result.predict(spells)
means = result.predict(spells, kind="mean")
rmst = result.predict(
    spells, jnp.array([1.0, 2.0]), kind="restricted_mean", grid=True,
)
inference = result.infer(covariance=ModelBased())
```

`predict(data)` defaults to the marginal median. A scalar value broadcasts over
observations; a vector is row-aligned. Pass `grid=True` with a one-dimensional
time or horizon array for an observation-by-value grid. Select
`kind="quantile"` with a probability value for other quantiles.
Conditional predictions accept an
explicit entry value for survival and cumulative hazard. Select the outcome
law explicitly; a substituted estimator must match it. Weibull and log-logistic use log shape, whereas the lognormal
ancillary parameter is log scale. Inference uses the observed-information
model-based covariance and returns its availability; it is conditional on the
model assumptions and does not establish finite-sample calibration.

Independent predictions accept `Observations`, named mappings and supported
eager Arrow-compatible frames through the same input adapter. Supply the retained
predictor roles and, for fixed effects, trained entity/period keys. Mapping
scalars broadcast; undeclared mapping columns are rejected. Frame predictions
select the required columns without relearning the fitted design.

Mean prediction is unconditional on entry and takes no time/grid argument.
Log-logistic shape at or below one has no finite mean, so its mean prediction is
NaN even when the fit and its other predictions are available. Its finite-horizon
restricted mean uses bounded quadrature; unresolved results are NaN. Spatial
survival stays lognormal and its mean uses the propagated node marginal variance.

For repeated spells, add entity and integer period columns and select effects:

```python
from statadic.api.data import PanelLayout

panel_spells = Observations(
    {
        "x": jnp.array([-1.0, -0.7, -0.2, 0.1, 0.3, 0.6, 0.9, 1.2]),
        "duration": jnp.array([0.8, 1.6, 0.9, 1.7, 2.4, 1.3, 2.1, 3.0]),
        "event": jnp.array([1, 1, 1, 0, 1, 1, 0, 1]),
        "unit": jnp.array([0, 1, 2, 3, 0, 1, 2, 3]),
        "period": jnp.array([0, 0, 0, 0, 1, 1, 1, 1]),
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

Effects may be `pooled`, `unit`, `time`, or `two-way`; coding and reference
levels are retained for prediction. Each entity-period has at most one spell,
while periods and entities can be unbalanced. These fixed effects do not add
frailty or model dependence between repeated spells. Unit, time and two-way
effects reject explicit `Intercept()`; use the effects' location component.
New fixed-effect levels
are rejected, and rank and fit diagnostics remain visible. Two-way coding uses
one time reference: disconnected unit–period incidence can leave dependent
columns and is rejected as rank deficient, even if each component could be
identified under a different normalization.

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

The [AFT transformation](formal/claims/STATADIC-SURVIVAL-002.md) and
[independent censoring with delayed entry](formal/claims/STATADIC-SURVIVAL-003.md)
state the observation-law assumptions. Their claim pages retain source locators
and human correspondence-review status. Numerical availability does not test
noninformative censoring or establish coverage.

The {download}`matched lifelines definitions <../tests/testing/test_survival_references.py>`
compare specified fits and predictions.
{ref}`Independent AFT <research-procedure-survival.independent>` and
{ref}`spatial survival <research-procedure-survival.spatial>` have separate
study mappings. The [known limits](validation/known-limitations.md) retain
cell-specific AFT results and high-censoring integration failures. Small
Gaussian controls do not qualify the full joint spatial-survival workload.

</details>

## Frozen structural contrasts

A structural contrast compares two predictor values in the fitted linear
predictor, using the **original training basis**. It does not relearn spline
knots, reselect categorical levels, or propagate through the spatial graph.

```python
from statadic.api.design import CyclicSpline
from statadic.api.inference import ModelBased

season = CyclicSpline("phase", df=3, period=1.0)
# Suppose spells also contains the numeric phase column.
seasonal_model = Regression(
    RightCensored("duration", "event"),
    Intercept() + Linear("x") + season,
    law=LogNormalAFT(),
)
fit = seasonal_model.estimate(spells).require_valid()
contrast = fit.design_contrast(season, at=0.25, versus=0.75)
inference = fit.infer(covariance=ModelBased())
result = inference.contrast(contrast, reference="normal")
print(result.estimate, result.standard_error, result.valid)
```

Compiled weights belong to the fitted design that produced them. To reuse the
same question across training samples, pass a
`RowDifference(at={"x": 1., "phase": .25}, versus={"x": 1., "phase": .75})`
from `statadic.api.design` to each inference result's `contrast` method. Each fit
evaluates both complete rows with its own learned transformation on the linear
predictor scale. A contrast result retains its target as a
`LinearPredictorContrast`, which describes the two predictor settings. Covariance and the reference distribution remain explicit.

The same `design_contrast` method is available on a fitted spatial survival
result. There its estimand is a **structural log-duration** contrast.
Graph-propagated marginal survival, median or duration effects require a
separate calculation and cannot be inferred from the structural contrast
alone. An identified point contrast may be retained even if its sampling
variance is unavailable; `ContrastResult.valid` and `unavailable_reason`
make that distinction explicit. The initial contrast operation supports a
normal reference only, selected by the caller.

## Spatial lag of log duration

`Regression(..., law=LogNormalAFT(), dependence=SpatialLag(...))` models endogenous log-time dependence on each
period's graph. `PanelData` keeps covariates for **all active graph
nodes** separate from observed spell outcomes. A graph can change population
between periods, or retain nodes whose outcomes are missing. Missing outcomes
are integrated out; they are not observed zero durations. Each graph explicitly
names numeric unit identities in its own order.

The following continues from the panel example. Keys, draw counts and period
order are explicit. Larger budgets may be required for censored or missing
outcomes; inspect the numerical diagnostics rather than treating this small
example as an accuracy guarantee.

```python
import jax

from statadic.api.spatial.graphs import PanelData, SpatialLag
from statadic.api.regression import Regression
from statadic.api.survival import SpatialSurvivalData, refine
from statadic.estimation.survival import SpatialSurvivalML
from statadic.estimation.survival.refinement import IntegrationRefinement
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign

nodes = Observations(
    {
        "x": panel_spells.numeric("x"),
        "unit": panel_spells.numeric("unit"),
        "period": panel_spells.numeric("period"),
    }
)
graph = prepare_spatial_weights(
    (jnp.ones((4, 4)) - jnp.eye(4)) / 3,
    unit_ids=jnp.arange(4),
)
graphs = ((0, graph), (1, graph))
training_draws = tuple(
    IntegrationDesign.from_key(jax.random.key(k), replicates=2, draws=16)
    for k in (1, 2)
)
validation_draws = tuple(
    IntegrationDesign.from_key(jax.random.key(k), replicates=2, draws=16)
    for k in (101, 102)
)
data = PanelData(
    nodes, PanelLayout("unit", "period"), {"neighbors": graphs}, panel_spells
)
spatial_model = Regression(
    RightCensored("duration", "event"),
    Intercept() + Linear("x"),
    structure=PanelLayout("unit", "period"),
    law=LogNormalAFT(),
    dependence=SpatialLag("neighbors"),
    estimator=SpatialSurvivalML(rho_fixed=0.0),
)
prepared = spatial_model.prepare(SpatialSurvivalData(data, training_draws))
spatial_result = spatial_model.estimate_prepared(prepared)
curves = spatial_result.predict(
    data, jnp.array([1.0, 2.0]), kind="survival", grid=True,
)
# Numerical refinement retains every attempt and its integration designs.
refined = refine(
    prepared,
    validation_integrations=validation_draws,
    config=IntegrationRefinement(initial_draws=16, max_draws=64, replicates=2),
)
# Inspect refined.refinement.attempts and refined.result.report()
# before drawing inferential conclusions.
print(spatial_result.report().to_text())
```

`rho_fixed=0.0` is a useful independent-error control; omit it to estimate the
spatial lag. Add `SpatialX("x", "neighbors")` to the terms for
named exogenous WX predictors. Spatial prediction returns unconditional
marginal node distributions, grouped by period and ordered by graph identity;
it does not condition on observed neighbours' durations. Spatial positive entry
is currently rejected by the host. Dense graph likelihood and integration are
required; spatial survival uses lognormal errors and constant ancillary scale.

A single fit uses the supplied fixed design, and its integration validity means
numerical representability, not certified accuracy. Independent keyed validation designs can be used by the numerical refinement controller. Fit and prediction preserve
caller-selected floating precision; the integration backend accepts float32 and
float64. `SpatialSurvivalData` owns the replayable draws; `SpatialSurvivalML`
owns numerical solver and integration controls. Host preparation, reports and
refinement are eager operations.
