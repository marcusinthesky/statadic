---
type: reference
description: Map earlier imports and calls to the current STATADIC API.
---

# Upgrading

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
    {"x": [0., 1., 2., 3.], "y": [1., 2., 2.8, 4.2]}
).require_valid()
```

## Description

Import a supported family from its owning module. The root package provides
version metadata, while compound namespaces guide you to their children. The
tables map earlier calls and paths to their current owners.

After changing an import, check the inputs and outputs against the reference for the current operation before interpreting its results.

For example, a regression model now estimates from prepared or raw data through its `estimate` methods, with uncertainty requested from the fitted result.

## Options

| Earlier call | Current interface |
| --- | --- |
| `PanelRegression`, `PanelOLS`, `Equation`, `Conditional`, `OLS`, `WLS`, `SurvivalRegression`, `DyadicRegression` | `Regression` with explicit terms, structure, law, effects and estimator |
| `PCA`, `PCoA` | `Ordination` with a Gram construction |
| `PanelLayout`, `SpatialPanelData` in `api.panel` | `api.data` |
| Panel `effects="fixed"` | `effects="unit"` |
| `predict(..., include_effects=True)` | `predict(..., kind="conditional")` |
| `model.fit`, `fit_prepared`, estimator `fit` | `estimate`, `estimate_prepared`, estimator `estimate` |
| `gmm_result.infer().fit` | `infer(covariance=GMMSandwich()).uncertainty.fit` |
| `inference.test(R, r, ...)` | `inference.test(Hypothesis(LinearRestriction(...)), ...)` |
| `iv_result.hansen_j(...)`, `gmm_result.hansen_j(...)` | `result.diagnose(HansenJ(...))` |
| Geometric `hypothesis=Reduced(...)` | `result.infer(resampling=...).test(Hypothesis(Reduced(...)))` |
| Implicit order in host BG, DW or HAC | `order_by="time"` or explicit `order_by=None` |
| GEL inference `.stat` | Request `GELCalibration` and test `OveridentifyingRestrictions` |
| `estimate(data, integrations=...)` | `estimate(SpatialSurvivalData(data, integrations))` |
| Survival prediction aliases | `predict` with `kind` and, for grids, `grid=True` |
| MMD or Energy fixed estimator and root options | Omit the fixed convention arguments |
| `Wasserstein1D()` | `Wasserstein1D(p=2)` |

## Results

The following paths identify the current owners of modules. Choose a function
or class from the relevant family for the operation you need.

| Earlier path | Current owner |
| --- | --- |
| `ground` | `geometry.ground` |
| `discrepancy.energy`, `discrepancy.balancing` | `discrepancy.mmd.energy`, its `balancing` child |
| Private `_mmd`, `discrepancy.mmd` | `discrepancy.mmd` |
| `discrepancy.exact`, private `_transport` | `discrepancy.transport.exact`, `transport.sliced` |
| Private `geometry._barycentre` | `geometry.barycentre` |
| `model.covariance` | `estimation.covariance` |
| Combined `model.dyadic` | `model.dyadic`, `estimation.dyadic`, `inference.dyadic` |
| `inference.gmm_covariance` | `inference.gmm.covariance` |
| `inference.bootstrap`, `permutation`, `block_length` | Children of `inference.resampling` |
| `decision.sharpe` | `estimation.sharpe`, `inference.sharpe` |
| `decision.equivalence`, `decision.multiplicity` | `inference.equivalence`, `inference.multiplicity` |
| `api.covariance` estimators | `Estimand(Covariance(), estimator=...)` and `estimation.covariance` |
| `geometry.ground.pairwise_distances` | `geometry.ground.metrics.cdist` or `pdist` |
| `core.statistics.Hypothesis`, `HypothesisT` | `core.statistics.Null`, `NullT` |
| `api.regression.RelativePrecision` | `api.data.RelativePrecision` |
| `api.regression.SpatialInformation` | `api.spatial.cross_section.SpatialInformation` |
| `api.regression.Model`, `Result`, `FitContext`, `PreparedRegression` | `api.base` |
| Regression covariance and inference types | `api.inference` |
| `api.spatial` result imports | Import the appropriate spatial family explicitly |
| `api.regression.PanelOLS`, panel result, layout and covariance | `Regression`, `api.panel` and `api.data` |

## Methods and formulas

| Earlier operation | Current operation |
| --- | --- |
| Linear `ols_fit`, `wls_fit` | `Regression` or `LeastSquares().estimate(LinearProblem(...))` |
| GLM `poisson_log_fit`, `binomial_logit_fit` | `Regression` with a law, or `PoissonMLE` / `BinomialLogitMLE` |
| Panel fit functions | `Regression` or `Pooled`, `Within`, `Between`, `FirstDifference`, `RandomEffects`, `FamaMacBeth` estimators |
| Panel `predict_panel` | `PanelResult.predict` |
| GMM and GEL fit functions | `Estimand(MomentConditions(), estimator=...)` or the matching numerical estimator |
| Spatial ML, SLX, IV and GMM fit functions | Spatial estimator classes. The compositional host does not yet provide spatial IV or GMM |
| Spatial panel ML and KKP fit functions | `Regression` with a panel layout or `PanelML` / `KKP` |
| Dyadic fit functions | `Regression` with `Dyads` or `DyadicEstimator`, followed by dyadic inference |
| Eager `pcoa`, `pcoa_matrix` | `Ordination(DistanceGram(), ...)` or `pcoa_kernel` |
| `ledoit_wolf_sample` | `Estimand(Covariance(), estimator=LedoitWolf())` |
| `disco_permutation_test` | Geometric regression with response-permutation inference, or the native DISCO null |
| `mantel_test` | `api.association.Mantel` with a hypothesis or the native permutation kernel |
| `permutation_test` | `inference.resampling.permutation.permutation_test_result` |
| Mantel and PERMDISP inference under geometry or association | `inference.mantel` and `inference.dispersion` |
| Raw matrix from a custom `GeometrySpec.matrix` | `HilbertianMeasureMatrix` with a producer warrant, or `None` |
| Separate statistic and metric in optimized energy permutation | `energy_geometry` binds ground and exponent before `energy_permutation_test` |
| String ground in numerical composition | Use a declared ground strategy. Parsing remains available at an eager boundary |

The empirical energy V-statistic can be metrized only when its construction
separates measures. A signed MMD U-statistic is an estimate, not a distance.
A geometric DISCO fit requires a separate inference request for its p-value.

## Assumptions and limits

Earlier class-instance pickles may require migration. Existing JSON, Parquet and numeric NPZ research artifacts do not require class-instance conversion. Imports do not promise universal pickle compatibility.

## Evidence

The [change log](../../CHANGELOG.md) records released changes, and tests for
individual workflows check the current interfaces.

## See also

- [Regression](regression.md)
- [Method catalogue](method-catalogue.md)
