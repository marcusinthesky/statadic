---
type: reference
description: Fit spatial regression, static panel and dyadic models with explicit graphs, effects and covariance requests.
---

# Spatial and panel models

## Usage

```python
import jax.numpy as jnp
import numpy as np
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.spatial.cross_section import SpatialInformation
from statadic.api.spatial.graphs import SpatialData, SpatialLag
from statadic.operators.spatial import prepare_spatial_weights

rng = np.random.default_rng(12)
units = 32
w = 0.7 * np.roll(np.eye(units), 1, axis=1) + 0.3 * np.roll(np.eye(units), -2, axis=1)
x = rng.normal(size=units)
y = np.linalg.solve(np.eye(units) - 0.2 * w, 0.4 + 0.8 * x + 0.3 * rng.normal(size=units))

data = SpatialData(
    Observations({"x": jnp.asarray(x), "y": jnp.asarray(y)}),
    {"neighbors": prepare_spatial_weights(w)},
)
model = Regression("y", Intercept() + Linear("x"), dependence=SpatialLag("neighbors"))
fit = model.estimate(data).require_valid()
report = fit.infer(covariance=SpatialInformation()).report(reference="normal")
print(fit.parameter_names)
print("\n".join(report.to_text().splitlines()[:5]))
```

## Description

{py:class}`~statadic.api.regression.Regression` fits spatial and panel models
when you add a graph-based `dependence`, a panel `structure` or both. Graph
preparation lives in `statadic.operators.spatial` and `statadic.operators.panel`,
so the inputs to a fit do not depend on estimation or inference code. Spatial
likelihoods, instrumental variables, GMM and the KKP component model live in
`statadic.estimation.spatial`, with covariance and diagnostics in
`statadic.inference.spatial`. General static panels use the estimators in
`statadic.estimation.panel` over a `PanelProblem`, and their covariance comes
from `statadic.inference.panel_covariance`.

## Options

`SpatialLag`, `SpatialError` and `SpatialX` name the graph that acts on the
outcome, on the error and on a predictor. Each graph is a prepared
`SpatialWeights` that preserves the edges and the unit order you supplied, with
normalization, isolates and the diagonal chosen explicitly.

| Model | Specification | Covariance request |
| --- | --- | --- |
| SAR, SEM, SDM, SDEM | `dependence=SpatialLag(g)` or `SpatialError(g)`, with `SpatialX` for WX | `SpatialInformation()` |
| Spatial panels | `PanelData` with `PanelLayout`, time-major | `SpatialInformation()` |
| Static panels | `structure=PanelLayout(...)`, `estimator=Within()` and others | `PanelCovariance(...)` |
| Fama-MacBeth | `estimator=FamaMacBeth()` | `FamaMacBethCovariance(...)` |

`Pooled`, `Within`, `Between`, `FirstDifference` and `RandomEffects` complete the
static panel estimators. Host effects (`effects="unit"`, `"time"` or
`"two-way"`) supply the location, so they reject an explicit `Intercept()`.

### Dyadic models

`Regression(..., structure=Dyads("i", "j"), effects=EndpointEffects("fe"))`
aligns outcomes with undirected endpoint identifiers and adds symmetric node
effects. Calling `estimate` returns a `DyadicResult`. Bind a node bootstrap with
`fit.infer(resampling=DyadicInference(bootstrap_iters=199, ...))`, and test one
coefficient against zero with `.test(Hypothesis(LinearRestriction.zero("x")))`.
Weights on dyads follow from products of endpoint counts, so they differ from the
row schedules in the [resampling reference](resampling.md).
Supply a `NodeCountSchedule` to pair the same node draws across analyses.
The supported test uses a two-sided add-one sign tail, and failed draws withhold
its calibrated p-value. Joint restrictions, nonzero targets and null-imposed
bootstrap tests require other procedures.

## Results

Spatial results name each parameter after the graph it belongs to, such as
`rho[neighbors]`, `lambda[neighbors]` and `W[neighbors]:x`, and reports and
restrictions use those names. Fits record solver status, rank, support and
availability separately, so an unidentified fit exposes its parameters as
unavailable.
`PanelFit.df_resid` gives residual degrees of freedom including absorbed
effects, and `FamaMacBethFit.df_resid` is the number of identified cross
sections minus one.

`PanelData(nodes=..., outcomes=..., panel=layout, graphs={...})` holds full node
covariates apart from observed outcomes, which lets populations change and
outcomes go missing. Prediction returns a `SpatialPanelPrediction` with period
IDs and one value array per period. Choose `kind="structural"` or
`"reduced-form"` where both exist. Saved graph artifacts predict against their
frozen graph from the training data, and `prepare_panel_sample` makes row and column removal explicit
for general panels.

## Methods and formulas

The marginal spatial panel model is $A_t y_t = X_t \beta + \alpha + B_t^{-1}
\varepsilon_t$ with an unfiltered entity effect $\alpha$. Random effects use a
Woodbury marginal likelihood, while KKP also filters the entity component
spatially, so the two are distinct specifications. For SAR and SEM, one spectrum
of the dense graph gives the log-Jacobian $\sum \log\lvert 1 - s\lambda_j(W)\rvert$
at each search point $s$. A balanced panel with a common graph profiles on
time-demeaned arrays and keeps all $T$ log-Jacobians.

## Assumptions and limits

- Conditioning on a graph does not justify learning that graph from the same
  outcomes. Full rank does not establish strong instruments, and lagging a
  constant under a row-stochastic $W$ gives none.
- Numeric unit keys and ordered period keys are array leaves, so map external
  labels first and reindex a changing unit order before fitting.
- `PanelML` needs a balanced design $(T, N, p)$, outcome $(T, N)$ and weights
  $(N, N)$ or $(T, N, N)$. Missing outcomes are not zeros.
- First differences need consecutive integer time keys, because they do not bridge gaps in the series.
- A variance component on the boundary of its range receives no ordinary joint inference.
- Row HC is not a spatial-error covariance. Robust LM adjusts for a competing
  spatial specification, not for heteroskedasticity.
- Exact likelihoods, trace diagnostics and quadratic-error GMM need dense
  graphs. Sparse input supports SLX, IV and GMRES solves.
- Dynamic panels, regimes, SUR and spatial GEL are outside the scope of this page.
- Wald intervals for SAR in small samples can under-cover, so treat them as
  approximate.

## Evidence

- Mathematical results: the {ref}`full-roster Jacobian <formal-STATADIC-SPATIAL-001>`,
  the {ref}`observed covariance <formal-STATADIC-SPATIAL-002>` and
  {ref}`two-way panel absorption <formal-STATADIC-PANEL-002>`.
- Studies: {ref}`point fits <research-procedure-spatial.point-estimation>`,
  {ref}`spatial panels <research-procedure-spatial.panel>` and
  {ref}`covariance and impacts <research-procedure-spatial.covariance-impacts>`.
- Reference comparisons: coefficients, likelihoods and covariance agree with
  spreg and linearmodels on matched graphs, designs and normalizations.
- Calibration of Gaussian SAR Wald intervals appears in the
  [known limitations](../validation/known-limitations.md).

## See also

- [Composition](composition.md) and [survival models](survival.md)
- [Learned panel graphs tutorial](../guides/learn-panel-graphs.py)
- [Reports](reporting.md)
