# Public modules and migration

STATADIC organizes implemented capabilities by scientific role, then family. Use
explicit owning-family imports. Root `statadic` exposes distribution version metadata;
root and compound namespaces document their children without eagerly loading
unrelated statistical methods. Family packages expose selected supported APIs.

`statadic.api` adds host workflows above these owners. Import the required family,
such as `statadic.api.regression` or `statadic.api.ordination`; its root remains a
navigation namespace. Model fitting uses public estimator classes; the migration
table records the removed function paths and their class replacements.
The [composition reference](reference/composition.md) explains identity, retained
preparation, qualified geometry and prediction artifacts.

A public module has no underscored import-path component and declares a literal
`__all__`; navigation namespaces use an empty list. A name is public only when
listed in a public module's `__all__`. Every public-looking
top-level function, class or type alias is listed or starts with an underscore.
Public signatures and catchable exceptions use public types. Cross-module
imports use public names, with visibility-scoped Tach interfaces for private
helpers. The inventory enforces declarations; Tach exact mode checks dependencies.

## Canonical families

The current API cleanup uses the following call migrations. Numerical components
remain accessible from their existing owners.

| Earlier call | Current call |
| --- | --- |
| `PanelRegression(..., layout=layout)` | `Regression(..., structure=layout, estimator=...)` |
| `PanelOLS(..., effects=FixedEffects(entity=True, time=False))` | `Regression(..., structure=layout, effects="unit")`; use `"time"`, `"two-way"` or `"pooled"` for the other combinations |
| `Equation`, `Conditional`, `OLS`, `WLS`, `SurvivalRegression`, `DyadicRegression` | `Regression` with top-level terms and explicit law, observation mechanism, precision or structure |
| `PCA`, `PCoA` model constructors | `Ordination` with a feature, kernel or distance Gram construction |
| `panel_result.predict(data, include_effects=True)` | `panel_result.predict(data, kind="conditional")`; the default remains `kind="xb"` |
| Balanced spatial panel `effects="fixed"` or result tag `"fixed"` | `effects="unit"` or result tag `"unit"` |
| `gmm_result.infer().fit` | `gmm_result.infer(covariance=GMMSandwich()).uncertainty.fit`; the bound inference also supplies `report`, `intervals` and `test` |
| Host `inference.test(R, r, ...)` | `inference.test(Hypothesis(LinearRestriction(...)), ...)`; native `linear_restriction_test` keeps array inputs |
| `iv_result.hansen_j(...)`, `gmm_result.hansen_j(...)` | `result.diagnose(HansenJ(...))`; native IV and GMM test kernels remain distinct |
| `hypothesis=Reduced(...)` in geometric inference | `result.infer(resampling=...).test(Hypothesis(Reduced(...)))` |
| Host BG, DW or HAC with implicit row order | Required `order_by="time"` or explicit `order_by=None` |
| `core.statistics.Hypothesis`, `HypothesisT` type axis | `core.statistics.Null`, `NullT`; the host value is `api.hypotheses.Hypothesis` |
| `gel_result.infer(...).stat` | `gel_result.infer(calibration=GELCalibration(...)).test(Hypothesis(OveridentifyingRestrictions(...))).evidence.stat` |
| Host model `estimate(data, integrations=...)` | `estimate(SpatialSurvivalData(panel_data, integrations))`; expert refinement uses `refine(prepared, ...)` |
| Survival prediction aliases | `result.predict(data)` for median, or explicit `kind="survival"`, `"mean"`, `"quantile"`, `"restricted_mean"` or `"cumulative_hazard"`; grids require `grid=True` |
| Structural effects/process arguments on numerical estimators | Put observation-aligned roles on `PanelProblem`, `SpatialProblem` or the corresponding family problem; estimators own numerical controls |
| `MMD(..., estimator="v", rooted=True)` / `Energy(..., estimator="v", rooted=True)` | Omit the fixed convention arguments |
| `Wasserstein1D()` | `Wasserstein1D(p=2)` |

The table below records the earlier module migration; its historical names do
not restore removed constructors such as `FixedEffects`.

| Earlier path | Canonical owner |
| --- | --- |
| `api.panel.PanelLayout`, `SpatialPanelData` | Shared aligned input carriers `api.data.PanelLayout`, `SpatialPanelData` |
| Result imports from `api.spatial` | Explicit `api.spatial.cross_section`, `panel`, `observed`, `graph_regression` or `prediction` owner; the compound namespace is a navigation map |
| `ground` | `geometry.ground` |
| `discrepancy.energy`, `discrepancy.balancing` | `discrepancy.mmd.energy`, its `balancing` child |
| `discrepancy.mmd`, private `_mmd` | Public `discrepancy.mmd` family |
| `discrepancy.exact`, private `_transport` | `discrepancy.transport.exact`, `transport.sliced` |
| Private `geometry._barycentre` | `geometry.barycentre`, with `mmd.energy` and `wasserstein` children |
| `model.covariance` | `estimation.covariance` |
| Combined `model.dyadic` | Specification `model.dyadic`, point fit `estimation.dyadic`, bootstrap `inference.dyadic` |
| `inference.gmm_covariance` | `inference.gmm.covariance`; tests in `gmm.hypothesis` |
| GEL fit-and-test entry points | `estimation.gel` fits, then `inference.gel.gel_test` |
| `inference.bootstrap`, `permutation`, `block_length` | Corresponding children of `inference.resampling` |
| `decision.sharpe` | Point estimate `estimation.sharpe`; uncertainty/tests `inference.sharpe` |
| `decision.equivalence`, `decision.multiplicity` | `inference.equivalence`, `inference.multiplicity` |
| Eager `ledoit_wolf_sample`, `LedoitWolfResult` | `Estimand(Covariance(), estimator=LedoitWolf())` with the brick from `estimation.covariance`, or its explicit kernel inside transforms |
| Eager `pcoa`, `pcoa_matrix` | `api.ordination.Ordination(DistanceGram(), ...)`, or `geometry.embedding.pcoa_kernel` |
| `geometry.ground.pairwise_distances` | `geometry.ground.metrics.cdist` (two samples) and `pdist` (one sample) |
| `disco_permutation_test`, `DiscoPermutationResult` | Geometric `Regression`, then `.infer(resampling=...).test(Hypothesis(Reduced(...)))`, or `inference.disco.disco_permutation_null` with `inference.calibration.add_one_p_kernel` |
| `mantel_test`, `MantelTestResult` | `api.association.Mantel(...).estimate(MantelInput(...)).infer(resampling=...).test(Hypothesis(MatrixExchangeability()))`, or the native Mantel permutation kernel |
| `permutation_test` | `inference.resampling.permutation.permutation_test_result`, reported through `to_host_test_result` |
| `inference.dyadic.fit_dyadic_model` | `Regression(..., structure=Dyads(...)).estimate(...).infer`, or `estimation.dyadic.DyadicEstimator().estimate(model)` followed by `inference.dyadic.bootstrap_dyadic` |
| `estimation.dyadic.fit_dyadic` | `Regression(..., structure=Dyads(...))`, or `estimation.dyadic.DyadicEstimator().estimate(model)` |
| `estimation.gmm.affine_gmm_fit`, free `api.moments.estimate` | `Estimand(MomentConditions(), estimator=IteratedGMM(...))`, or `estimation.gmm.IteratedGMM` / `FixedWeightGMM` over `estimation.moments.AffineMomentProblem` |
| `estimation.gel.fixed_gel_fit`, `affine_gel_fit`, old host GEL model constructors | `Estimand(MomentConditions(), estimator=FixedGEL(...))` / `AffineGEL(...)`, or those same numerical bricks directly |
| `estimation.spatial.spatial_ml_fit`, `spatial_slx_fit` | `Regression` with named dependence or `SpatialX`, or low-level `estimation.spatial.ML` / `SLXLeastSquares` |
| `estimation.spatial.spatial_iv_fit`, `spatial_gmm_fit` | Low-level `estimation.spatial.SpatialIV` / `SpatialGMM`; no compositional IV/GMM host yet |
| `estimation.spatial.spatial_panel_ml_fit`, `kkp_panel_fit` | `Regression` with `structure=PanelLayout(...)`, or low-level `estimation.spatial.PanelML` / `KKP` over `SpatialPanelProblem` |
| `api.covariance.SampleCovariance`, `LedoitWolf` | `Estimand(Covariance(), estimator=SampleCovariance(...))` / `LedoitWolf()`; numerical bricks live in `estimation.covariance`, returned results retain `.sigma` and shrinkage diagnostics |
| Custom `GeometrySpec.matrix` returning a raw `DMat` | Return `HilbertianMeasureMatrix(native, construction="...")` with an explicit producer warrant, or `None` for geometry without that capability |
| Public `estimation.panel.predict_panel` | Private helper behind `api.panel.PanelResult.predict` |
| `estimation.panel.pooled_panel_fit` | `Regression(..., structure=layout, estimator=Pooled())`, or brick `estimation.panel.Pooled().estimate(PanelProblem(...))` |
| `estimation.panel.fixed_effects_fit` | `Regression(..., structure=layout, estimator=Within(...))`, or brick `estimation.panel.Within(...).estimate(PanelProblem(...))` |
| `estimation.panel.between_fit` | `Regression(..., structure=layout, estimator=Between(reweight=...))`, or brick `estimation.panel.Between(...).estimate(PanelProblem(...))` |
| `estimation.panel.first_difference_fit` | `Regression(..., structure=layout, estimator=FirstDifference())`, or brick `estimation.panel.FirstDifference().estimate(PanelProblem(...))` |
| `estimation.panel.random_effects_fit` | `Regression(..., structure=layout, estimator=RandomEffects(...))`, or brick `estimation.panel.RandomEffects(...).estimate(PanelProblem(...))` |
| `estimation.panel.fama_macbeth_fit` | `Regression(..., structure=layout, estimator=FamaMacBeth())`, or brick `estimation.panel.FamaMacBeth().estimate(PanelProblem(...))` |
| `api.regression.PanelOLS`, `PanelResult`, `PanelLayout`, `FixedEffects`, `PanelCovariance` | Model construction uses `Regression`; results and covariance requests belong to `api.panel`, layout to `api.data`; effects use explicit strings |
| `estimation.linear.ols_fit`, `wls_fit` | `Regression` with optional relative precision, or brick `estimation.linear.LeastSquares().estimate(LinearProblem(...))` |
| `estimation.glm.poisson_log_fit` | `Regression(Observed(...), terms, law=Poisson())`, or brick `estimation.glm.PoissonMLE().estimate(PoissonProblem(...))` |
| `estimation.glm.binomial_logit_fit` | `Regression(Observed(...), terms, law=Binomial())`, or brick `estimation.glm.BinomialLogitMLE().estimate(BinomialProblem(...))` |
| `api.regression.Model`, `Result` | Workflow contracts `api.base.Model`, `Result`; extend a fitted term or native `Estimator` for existing regression compositions |
| `api.regression.FitContext`, `PreparedRegression` | Same names in `api.base` |
| `api.regression.CovarianceSpec`, `ModelBased`, `HC`, `HAC`, `Clustered`, `InferenceResult` | Same names in `api.inference` |
| `model.fit(data)`, `model.fit_prepared(prepared)`, `brick.fit(problem)` | `model.estimate(data)`, `model.estimate_prepared(prepared)`, `brick.estimate(problem)`; no class keeps `fit` |
| `api.regression.RelativePrecision` | `api.data.RelativePrecision` |
| `api.regression.SpatialInformation` | `api.spatial.cross_section.SpatialInformation` for compatible spatial ML fits |
| Mantel testing under association | Pure statistic `association.mantel`, testing `inference.mantel` |
| PERMDISP inference under geometry | Median geometry `geometry.median`, statistic `association.dispersion`, inference `inference.dispersion` |

The former `decision` and empty `experimental` roots are removed. These are
intentional clean import breaks, without compatibility facade trees or runtime
qualified-name mutation. Existing external pickles naming former modules may
require migration; governed research artifacts use JSON, Parquet and numeric NPZ
and do not require class-instance conversion. No universal pickle compatibility
is promised by a successful source-import migration.
Linear and GLM workflow classes now have implementation-qualified names under
`api.regression._linear` and `api.regression._glm`; keep using the curated
`api.regression` imports and treat serialized class identities as a migration
boundary.

## Choose the mathematical object

Energy's distance-induced MMD specialization retains its direct efficient
reductions. Signed U-statistic MMD estimates are not rootable distance matrices.
Exact and sliced transport remain different algorithms and domains; variant
exception classes remain available from their variant modules.

Covariance of observations, deterministic covariance operators and sampling
covariance of coefficients have different owners. Model specifications do not
fit parameters or run bootstrap inference. Fits retain actual observations,
parameter order, numerical evidence and the weighting state used in the solve.
An uncertainty owns its actual fit; shared intervals and restrictions consume
its structural capabilities. Static sample capacity is distinct from actual
contribution count, including partially available Fama–MacBeth periods.
`api.hypotheses` owns mathematical claims and host `TestOutcome` reports;
`api.diagnostics` composes family-supported procedures with retained fits.
No runtime outcome claims formal source approval. See [hypotheses](reference/hypotheses.md).

Resampling schedules record their design explicitly. Dyadic node-count schedules
are not row-index schedules. Relocation preserves random-key labels, retained
failures, solver checks and reference assumptions.

## Progressive disclosure

Each public initializer explains purpose, inputs/outputs, child selection and the
next composition step. Detailed derivations and precision/transform contracts
remain in the owning module and the specialized guides. Import a family such as
`from statadic.estimation.linear import LeastSquares, LinearProblem`, then its matching covariance and
`statadic.inference.parameters` procedures. `statadic.discrepancy.mmd` consistently names
the family module; import its `mmd` callable from that family.

The package `ARCHITECTURE.md` at the package root explains dependency direction;
the [question index](methods.md) connects complete workflows and the
[capability guide](capabilities.md) records transformation limits.
