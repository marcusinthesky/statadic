# Enforcement verification

## Integrated enforcement and shared policy

The export checkpoint passed before enforcement began. The coordinator enabled
Tach exact mode and removed the two measured unused api.inputs dependencies
(core and sample). Six scoped helper interfaces cover the reviewed renames. The
old studies projection exception is replaced with visibility restricted to GMS
and Wolak, which publicly re-export the catchable exception. The obsolete dyadic
fit public-interface pattern was removed with the retired module.

A separate read-only reviewer checked all 28 scoped interfaces: every exposed
name/visibility pair matches an actual direct import, including the three
source-identity `statadic.__file__` attribute consumers. The other 57 interfaces
retain uniform alphabetic-name patterns. No remaining private exception needed
removal or narrower visibility. Exact Tach validation passed. The independent
catalogue audit checked all 2,104 identity/target/method mappings with zero
mismatches. Documentation now states explicitly that only names listed in a
public module's `__all__` are public.

The inventory worker's first implementation passed 35 registry tests and the
read-only current-snapshot check. Independent review requested strict rejection
of augmented public export declarations and reporting unlisted definitions even
when their module lacks a literal declaration. These corrections precede final
acceptance; their results are recorded below when verified. An initial worker
registry attempt failed formatting, was explicitly formatted, then passed.

## Consumer impact

## Consumer import audit

AST audit of Python files in each stated tree (including tests and examples), against the final inventory and `git show HEAD:src/lean/registry/public-api.json`. Paths below are relative to each scope root. Names are deduplicated per module; each entry gives one actual file and line as evidence. Public child-module imports are listed separately. The final inventory has 2,104 identities.

### pricing-perspective

Scope `/home/marcussky/Git/pricing-perspective/src/python`: 487 Python files parsed; 47 modules and 110 distinct module/name imports; 1 distinct child modules; 1 plain import statements. Parse errors: 0. Final-inventory missing/private imports: none. Imports absent from HEAD inventory: none.

- `statadic.api.association`: `Mantel` (`apps/pipeline/src/pipeline/stages/papers/paper1/etnic_alignment.py:28`).
- `statadic.api.comparison`: `PERMANOVA` (`apps/pipeline/src/pipeline/stages/papers/paper1/w2_permanova.py:10`); `PERMDISP` (`apps/pipeline/src/pipeline/stages/papers/paper1/w2_permanova.py:10`); `PermutationPlan` (`apps/pipeline/src/pipeline/stages/papers/paper1/etnic_alignment.py:29`).
- `statadic.api.covariance`: `LedoitWolf` (`apps/pipeline/src/pipeline/stages/papers/paper5/economic_value.py:19`).
- `statadic.api.data`: `Observations` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:8`).
- `statadic.api.dyadic`: `DyadicRegression` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:9`).
- `statadic.api.embedding`: `PCoA` (`apps/pipeline/src/pipeline/stages/papers/paper1/pcoa.py:17`); `PCoASpectrum` (`apps/pipeline/src/pipeline/stages/papers/paper1/w2_permanova.py:21`).
- `statadic.association.mantel`: `MantelMethod` (`apps/pipeline/src/pipeline/stages/papers/paper1/etnic_alignment.py:35`).
- `statadic.core.equations`: `PlainLongRun` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:24`).
- `statadic.core.random`: `cell_key` (`apps/pipeline/tests/papers/paper5/estimator_validation.py:30`).
- `statadic.core.typing`: `Array` (`apps/hotelling/src/hotelling/dynamics/engine.py:15`); `ArrayLike` (`apps/hotelling/src/hotelling/dynamics/engine.py:15`); `Bool` (`apps/hotelling/src/hotelling/dynamics/valuation.py:20`); `Float` (`apps/hotelling/src/hotelling/dynamics/engine.py:15`); `Int` (`apps/hotelling/src/hotelling/dynamics/engine.py:15`); `PRNGKey` (`apps/hotelling/src/hotelling/dynamics/observation.py:7`).
- `statadic.discrepancy.metrize`: `NegativeEnergyFunctionalError` (`apps/pipeline/src/pipeline/stages/substrate/energy_metric.py:8`); `NonfiniteEnergyFunctionalError` (`apps/pipeline/src/pipeline/stages/substrate/energy_metric.py:8`); `sqrt_energy_functional` (`apps/pipeline/src/pipeline/stages/substrate/energy_metric.py:8`).
- `statadic.discrepancy.mmd`: `BUILTIN_MMD_KERNELS` (`apps/pipeline/src/pipeline/_kernels/typed_geometry.py:15`); `mmd_matrix_values` (`apps/pipeline/src/pipeline/_kernels/typed_distances.py:17`); `mmd_squared_matrix` (`apps/pipeline/src/pipeline/_kernels/typed_distances.py:17`).
- `statadic.discrepancy.mmd.energy`: `empirical_mean_ground_distances` (`apps/pipeline/src/pipeline/_kernels/typed_geometry.py:16`).
- `statadic.discrepancy.mmd.energy.matrix`: `energy_functional_matrix` (`apps/pipeline/src/pipeline/_kernels/typed_distances.py:18`).
- `statadic.discrepancy.transport.exact`: `balanced_wasserstein1_from_cost` (`apps/pipeline/tests/kernels/test_wasserstein.py:9`); `balanced_wasserstein_from_cost` (`apps/pipeline/tests/test_typed_kernels.py:9`); `balanced_wasserstein_from_cost_jax` (`apps/pipeline/src/pipeline/_kernels/typed_distances.py:19`); `chord_cost_matrix` (`apps/pipeline/src/pipeline/stages/papers/paper1/wasserstein_comparator.py:21`); `exact_balanced_wasserstein1` (`apps/pipeline/tests/kernels/test_wasserstein.py:13`); `exact_balanced_wasserstein_jax` (`apps/pipeline/src/pipeline/_kernels/typed_distances.py:19`).
- `statadic.discrepancy.transport.univariate`: `wasserstein_1d_power` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:26`).
- `statadic.estimation.covariance.estimators`: `sample_covariance_kernel` (`apps/pipeline/src/pipeline/stages/papers/paper3/representation_ablation.py:13`).
- `statadic.estimation.dyadic`: `DyadicComputationError` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/estimation.py:10`); `DyadicEstimator` (`apps/pipeline/tests/papers/paper1/test_dyadic_confound.py:12`); `build_dyadic_design` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/diagnostics.py:9`).
- `statadic.estimation.glm`: `BinomialLogitMLE` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:27`); `BinomialProblem` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:27`); `PoissonMLE` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:27`); `PoissonProblem` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:27`).
- `statadic.estimation.gmm`: `FixedWeightGMM` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:33`); `IteratedGMM` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:33`).
- `statadic.estimation.linear`: `LeastSquares` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:34`); `LinearProblem` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:34`).
- `statadic.estimation.moments`: `AffineMomentProblem` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:38`).
- `statadic.estimation.panel`: `PanelProblem` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:39`); `Within` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:39`).
- `statadic.estimation.spatial`: `ML` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:40`); `SAR` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:40`); `SLXLeastSquares` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:40`); `SpatialProblem` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:40`).
- `statadic.estimation.spatial.panel`: `spatial_panel_profile` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:41`).
- `statadic.geometry.barycentre.mmd`: `kernel_barycentre_weights` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/mmd.py:14`).
- `statadic.geometry.barycentre.mmd.energy`: `EnergyBarycentreDiagnostics` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/_diagnostics.py:16`); `energy_barycentre_weights` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/energy.py:16`).
- `statadic.geometry.barycentre.wasserstein`: `WassersteinBarycentreResult` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/wasserstein/measure.py:37`); `solve_wasserstein_free_support_barycentre` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/wasserstein/measure.py:17`); `solve_wasserstein_measure_barycentre` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/wasserstein/measure.py:20`).
- `statadic.geometry.ground.config`: `resolve_ground_distance` (`apps/pipeline/src/pipeline/_kernels/typed_geometry.py:17`).
- `statadic.geometry.ground.metrics`: `cdist` (`apps/pipeline/src/pipeline/_kernels/typed_geometry.py:18`); `squared_euclidean_distance` (`apps/hotelling/src/hotelling/dynamics/geometry.py:14`).
- `statadic.geometry.ground.similarities`: `RBFKernel` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:42`); `gram` (`apps/pipeline/src/pipeline/_kernels/typed_geometry.py:19`); `rbf_kernel` (`apps/pipeline/src/pipeline/_kernels/typed_geometry.py:19`).
- `statadic.geometry.neighbour`: `FloatMatrix` (`apps/pipeline/src/pipeline/_kernels/neighbour_stability.py:31`); `NEIGHBOUR_INPUT_AXIOMS` (`apps/pipeline/src/pipeline/_kernels/neighbour_stability.py:31`); `cross_encoder_neighbour_stability` (`apps/pipeline/src/pipeline/_kernels/neighbour_stability.py:32`); `neighbour_order` (`apps/pipeline/src/pipeline/_kernels/neighbour_stability.py:31`).
- `statadic.inference.dyadic`: `DyadicBootstrapDesign` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/estimation.py:11`); `DyadicBootstrapResult` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/estimation.py:20`); `DyadicFitRequest` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/classification.py:11`); `DyadicInference` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/classification.py:24`); `NodeCountSchedule` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:14`); `bootstrap_coefficient_inference` (`apps/pipeline/src/pipeline/stages/papers/paper1/dyadic_confound.py:19`); `bootstrap_dyadic` (`apps/pipeline/tests/papers/paper1/test_dyadic_confound.py:13`); `draw_node_counts` (`apps/pipeline/tests/papers/paper1/test_dyadic_confound.py:13`); `node_bootstrap_coefficient` (`apps/pipeline/tests/papers/paper1/test_dyadic_confound.py:13`).
- `statadic.inference.glm_covariance`: `GLMUncertainty` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:73`); `glm_covariance` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:43`).
- `statadic.inference.gmm.covariance`: `gmm_covariance` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:44`).
- `statadic.inference.linear_covariance`: `classical_covariance` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:45`).
- `statadic.inference.parameters`: `coefficient_intervals` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:46`).
- `statadic.inference.resampling.block_length`: `optimal_block_length_numpy` (`apps/pipeline/src/pipeline/stages/papers/paper3/representation_ablation.py:14`).
- `statadic.inference.resampling.schedule`: `ReplicateStatus` (`apps/pipeline/src/pipeline/stages/papers/paper3/representation_ablation.py:15`); `ReplicateValue` (`apps/pipeline/src/pipeline/stages/papers/paper3/representation_ablation.py:15`); `evaluate_replicates` (`apps/pipeline/src/pipeline/stages/papers/paper3/representation_ablation.py:15`); `iid_row_schedule` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:47`); `linear_pairs_replicates` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:47`); `percentile_summary` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:47`); `refit_replicates` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:47`); `row_resample_schedule` (`apps/pipeline/src/pipeline/stages/papers/paper3/representation_ablation.py:15`).
- `statadic.model.dyadic`: `DyadicInputError` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/classification.py:12`); `DyadicModel` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:10`); `DyadicSample` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:15`).
- `statadic.model.glm`: `BinomialLogitEquation` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:74`); `PoissonLogEquation` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:74`).
- `statadic.operators.design`: `NodeIndex` (`apps/pipeline/tests/papers/paper1/test_dyadic_confound.py:21`); `drop_collinear_columns` (`apps/pipeline/src/pipeline/stages/papers/paper1/dyadic_confound.py:26`); `fwl_coefficient` (`apps/pipeline/src/pipeline/stages/papers/paper1/dyadic_confound.py:26`); `fwl_residualize` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:11`); `symmetric_dyadic_effects` (`apps/pipeline/src/pipeline/stages/papers/paper1/_dyadic/adapters.py:11`).
- `statadic.operators.panel`: `prepare_panel_index` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:54`).
- `statadic.operators.spatial`: `prepare_spatial_panel_weights` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:55`); `prepare_spatial_weights` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:55`); `spatial_lag` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:55`); `spatial_solve` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:55`).
- `statadic.optimize.simplex`: `simplex_qp_result` (`apps/pipeline/src/pipeline/stages/papers/paper3/certificate_validation.py:16`).
- `statadic.sample.packed`: `PackedClouds` (`apps/pipeline/src/pipeline/_kernels/typed_barycentre/energy.py:17`).
- `statadic.sample.reduction`: `Projection` (`apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py:63`).

Child modules:

- `statadic.geometry.neighbour`: `apps/pipeline/tests/kernels/test_kernel_promotion_shims.py:19`.

Plain module imports:

- `statadic` (`apps/pipeline/src/pipeline/stages/papers/paper5/spatial_covariance.py:21`).

### statadic-publishing

Scope `/home/marcussky/Git/statadic/src/python/packages/statadic-publishing`: 12 Python files parsed; 6 modules and 11 distinct module/name imports; 0 distinct child modules; 0 plain import statements. Parse errors: 0. Final-inventory missing/private imports: none. Imports absent from HEAD inventory: none.

- `statadic.api.artifacts`: `load_analysis` (`tests/test_contracts.py:153`); `save_analysis` (`examples/ols-hc3/analysis.py:10`).
- `statadic.api.design`: `Intercept` (`examples/ols-hc3/analysis.py:11`); `Linear` (`examples/ols-hc3/analysis.py:11`).
- `statadic.api.inference`: `HC` (`examples/ols-hc3/analysis.py:12`).
- `statadic.api.plots`: `residual_plot` (`examples/ols-hc3/analysis.py:13`).
- `statadic.api.regression`: `OLS` (`examples/ols-hc3/analysis.py:14`).
- `statadic.api.reporting`: `ParameterRow` (`tests/test_contracts.py:154`); `Report` (`tests/test_contracts.py:154`); `freeze_metadata` (`src/statadic_publishing/manifest.py:8`); `metadata_to_json` (`src/statadic_publishing/manifest.py:8`).

### statadic-evidence

Scope `/home/marcussky/Git/statadic/src/python/packages/statadic-evidence`: 2 Python files parsed; 0 modules and 0 distinct module/name imports; 0 distinct child modules; 0 plain import statements. Parse errors: 0. Final-inventory missing/private imports: none. Imports absent from HEAD inventory: none.

### apps

Scope `/home/marcussky/Git/statadic/src/python/apps`: 153 Python files parsed; 84 modules and 213 distinct module/name imports; 2 distinct child modules; 3 plain import statements. Parse errors: 0. Final-inventory missing/private imports: none. Imports absent from HEAD inventory: [('statadic.inference.gms', 'ProjectionUnavailableError'), ('statadic.inference.mantel', 'mantel_bootstrap_kernel')].

- `statadic.api.artifacts`: `load_analysis` (`docs/tests/test_reporting_workflow.py:14`); `load_predictor` (`docs/tests/test_reporting_workflow.py:14`).
- `statadic.api.association`: `Mantel` (`studies/src/statadic_studies/adapters/groups.py:5`).
- `statadic.api.comparison`: `DISCO` (`studies/src/statadic_studies/adapters/groups.py:6`); `PERMANOVA` (`studies/src/statadic_studies/adapters/groups.py:6`); `PERMDISP` (`studies/src/statadic_studies/adapters/groups.py:6`); `PermutationPlan` (`studies/src/statadic_studies/adapters/groups.py:6`).
- `statadic.api.data`: `MeasureColumn` (`docs/src/statadic_docs_labs/native.py:16`); `Observations` (`docs/src/statadic_docs_labs/native.py:16`).
- `statadic.api.design`: `DistanceToLandmarks` (`docs/src/statadic_docs_labs/native.py:17`); `Intercept` (`docs/src/statadic_docs_labs/native.py:17`); `Linear` (`docs/src/statadic_docs_labs/native.py:17`).
- `statadic.api.embedding`: `PCA` (`studies/src/statadic_studies/adapters/population.py:10`); `PCoA` (`docs/src/statadic_docs_labs/native.py:18`).
- `statadic.api.inference`: `HC` (`docs/src/statadic_docs_labs/native.py:19`); `ModelBased` (`docs/src/statadic_docs_labs/native.py:19`).
- `statadic.api.regression`: `OLS` (`docs/src/statadic_docs_labs/native.py:20`).
- `statadic.api.reporting`: `ParameterRow` (`docs/scripts/compile_report.py:8`); `PlotSpec` (`docs/scripts/compile_report.py:69`); `ReferenceLine` (`docs/tests/test_reporting_workflow.py:15`); `Report` (`docs/scripts/compile_report.py:8`).
- `statadic.api.representation`: `Energy` (`docs/src/statadic_docs_labs/native.py:21`); `MMD` (`docs/src/statadic_docs_labs/native.py:21`); `Wasserstein1D` (`docs/src/statadic_docs_labs/native.py:21`).
- `statadic.core.equations`: `MeanCenteredLongRun` (`studies/src/statadic_studies/adapters/estimators.py:15`); `OneWayCluster` (`studies/src/statadic_studies/adapters/extended.py:137`); `PlainLongRun` (`studies/src/statadic_studies/adapters/extended.py:137`).
- `statadic.discrepancy.mmd`: `mmd_squared` (`performance/src/statadic_performance/workloads.py:10`).
- `statadic.discrepancy.mmd.energy`: `energy_distance` (`docs/src/statadic_docs_labs/native.py:22`); `energy_test_statistic` (`performance/history/benchmark_jit.py:31`); `mixture_energy_distance_from_blocks` (`studies/src/statadic_studies/adapters/discrepancy/mixture_bootstrap.py:6`).
- `statadic.discrepancy.mmd.energy.balancing.mixture`: `energy_distance_kernel_mixture` (`studies/src/statadic_studies/adapters/discrepancy/fixed_mixture.py:7`).
- `statadic.discrepancy.mmd.energy.balancing.weights`: `kernel_balancing_weights` (`studies/src/statadic_studies/adapters/resampling.py:14`).
- `statadic.discrepancy.mmd.energy.geometry`: `resolve_energy_geometry` (`docs/src/statadic_docs_labs/native.py:23`).
- `statadic.discrepancy.mmd.energy.kernel`: `distance_induced_kernel` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:11`).
- `statadic.discrepancy.mmd.energy.mixture`: `mixture_energy_distance` (`studies/src/statadic_studies/adapters/discrepancy/fixed_mixture.py:8`).
- `statadic.discrepancy.mmd.energy.scalar`: `energy_distance` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:12`).
- `statadic.discrepancy.mmd.energy.weighted`: `weighted_energy_cross` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:13`).
- `statadic.discrepancy.mmd.scalar`: `cosine_mean_mmd` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:14`); `mmd` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:14`); `mmd_squared` (`studies/src/statadic_studies/adapters/discrepancy/finite_bias.py:10`).
- `statadic.discrepancy.mmd.weighted`: `weighted_mmd_cross` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:15`).
- `statadic.discrepancy.transport`: `ProjectionDesign` (`studies/src/statadic_studies/adapters/population.py:11`); `exact_balanced_wasserstein1` (`studies/src/statadic_studies/adapters/population.py:11`); `exact_balanced_wasserstein2` (`studies/src/statadic_studies/adapters/population.py:11`); `sliced_wasserstein` (`studies/src/statadic_studies/adapters/population.py:11`); `sliced_wasserstein_under_design` (`studies/src/statadic_studies/adapters/population.py:11`); `wasserstein_1d` (`studies/src/statadic_studies/adapters/population.py:11`); `wasserstein_1d_power` (`studies/src/statadic_studies/adapters/population.py:11`).
- `statadic.discrepancy.transport.matrix`: `transport_cross` (`studies/src/statadic_studies/adapters/population.py:20`).
- `statadic.estimation.covariance`: `ledoit_wolf_sample_kernel` (`studies/src/statadic_studies/adapters/covariance.py:10`); `sample_covariance_kernel` (`studies/src/statadic_studies/adapters/covariance.py:10`).
- `statadic.estimation.covariance.cross_validation`: `cv_lambda_grid` (`studies/src/statadic_studies/adapters/covariance_tuning.py:7`).
- `statadic.estimation.covariance.estimators`: `sample_covariance` (`studies/src/statadic_studies/adapters/covariance_tuning.py:8`).
- `statadic.estimation.covariance.shrinkage`: `optimal_lambda_dist` (`studies/src/statadic_studies/adapters/covariance_tuning.py:9`); `plug_in_lambda_dist` (`studies/src/statadic_studies/adapters/covariance_tuning.py:9`); `sample_cov_entry_variance` (`studies/src/statadic_studies/adapters/covariance_tuning.py:9`); `shrink_to_dist` (`studies/src/statadic_studies/adapters/covariance_tuning.py:9`).
- `statadic.estimation.dyadic`: `DyadicComputationError` (`studies/src/statadic_studies/adapters/bounds.py:8`); `DyadicEstimator` (`studies/src/statadic_studies/adapters/bounds.py:8`).
- `statadic.estimation.gel`: `AffineGEL` (`studies/src/statadic_studies/adapters/moments.py:9`); `FixedGEL` (`studies/src/statadic_studies/adapters/estimators.py:16`).
- `statadic.estimation.glm`: `BinomialLogitMLE` (`studies/src/statadic_studies/adapters/estimators.py:17`); `BinomialProblem` (`studies/src/statadic_studies/adapters/estimators.py:17`); `PoissonMLE` (`studies/src/statadic_studies/adapters/estimators.py:17`); `PoissonProblem` (`studies/src/statadic_studies/adapters/estimators.py:17`).
- `statadic.estimation.gmm`: `FixedWeightGMM` (`studies/src/statadic_studies/adapters/estimators.py:23`); `IteratedGMM` (`studies/src/statadic_studies/adapters/estimators.py:23`).
- `statadic.estimation.iv`: `IVProblem` (`studies/src/statadic_studies/adapters/extended.py:138`); `IteratedIVGMM` (`studies/src/statadic_studies/adapters/extended.py:138`); `TwoStageLeastSquares` (`studies/src/statadic_studies/adapters/extended.py:138`); `TwoStepIVGMM` (`studies/src/statadic_studies/adapters/extended.py:138`).
- `statadic.estimation.likelihood`: `EstimatedDispersion` (`studies/src/statadic_studies/adapters/extended.py:20`); `FixedDispersion` (`studies/src/statadic_studies/adapters/extended.py:20`); `GammaMLE` (`studies/src/statadic_studies/adapters/extended.py:20`); `LikelihoodProblem` (`studies/src/statadic_studies/adapters/extended.py:20`); `NegativeBinomialMLE` (`studies/src/statadic_studies/adapters/extended.py:20`); `ProbitMLE` (`studies/src/statadic_studies/adapters/extended.py:20`).
- `statadic.estimation.linear`: `LeastSquares` (`performance/src/statadic_performance/workloads.py:12`); `LinearProblem` (`performance/src/statadic_performance/workloads.py:12`).
- `statadic.estimation.moments`: `AffineMomentProblem` (`studies/src/statadic_studies/adapters/estimators.py:24`); `FixedMomentProblem` (`studies/src/statadic_studies/adapters/estimators.py:24`).
- `statadic.estimation.panel`: `Between` (`studies/src/statadic_studies/adapters/dependence/linear_panel.py:9`); `FamaMacBeth` (`studies/src/statadic_studies/adapters/dependence/covariance.py:9`); `FirstDifference` (`studies/src/statadic_studies/adapters/dependence/linear_panel.py:9`); `PanelProblem` (`studies/src/statadic_studies/adapters/dependence/covariance.py:9`); `Pooled` (`studies/src/statadic_studies/adapters/dependence/linear_panel.py:9`); `RandomEffects` (`studies/src/statadic_studies/adapters/dependence/linear_panel.py:9`); `Within` (`studies/src/statadic_studies/adapters/dependence/covariance.py:9`).
- `statadic.estimation.sharpe`: `sample_skew_kurtosis` (`studies/src/statadic_studies/adapters/sharpe_moments.py:7`); `sharpe_ratio` (`studies/src/statadic_studies/adapters/sharpe_point.py:7`).
- `statadic.estimation.spatial`: `KKP` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:8`); `ML` (`studies/src/statadic_studies/adapters/dependence/spatial.py:8`); `ObservedPanelML` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:8`); `ObservedSpatialPanelProblem` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:8`); `PanelML` (`studies/src/statadic_studies/adapters/dependence/spatial_panel.py:8`); `SAC` (`studies/src/statadic_studies/adapters/dependence/spatial_durbin.py:8`); `SAR` (`studies/src/statadic_studies/adapters/dependence/spatial.py:8`); `SEM` (`studies/src/statadic_studies/adapters/dependence/spatial_durbin.py:8`); `SLXLeastSquares` (`studies/src/statadic_studies/adapters/dependence/spatial.py:8`); `SpatialGMM` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:8`); `SpatialIV` (`studies/src/statadic_studies/adapters/dependence/spatial_diagnostics.py:9`); `SpatialPanelProblem` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:8`); `SpatialProblem` (`studies/src/statadic_studies/adapters/dependence/spatial.py:8`).
- `statadic.estimation.survival`: `SpatialSurvivalML` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:7`); `SpatialSurvivalProblem` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:7`); `SurvivalML` (`studies/src/statadic_studies/adapters/estimators.py:25`); `SurvivalProblem` (`studies/src/statadic_studies/adapters/estimators.py:25`); `prepare_spatial_survival_period` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:7`); `spatial_survival_log_likelihood` (`studies/tests/dependence/test_spatial_survival_pilot.py:6`).
- `statadic.estimation.survival.refinement`: `IntegrationRefinement` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:12`); `refine_spatial_survival` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:12`).
- `statadic.geometry.barycentre.mmd.energy`: `energy_barycentre_weights_batched` (`studies/src/statadic_studies/adapters/resampling.py:16`).
- `statadic.geometry.ground.metrics`: `angular_distance` (`docs/src/statadic_docs_labs/native.py:24`); `euclidean_distance` (`docs/src/statadic_docs_labs/native.py:24`).
- `statadic.geometry.ground.similarities`: `COSINE_KERNEL` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:16`); `LINEAR_KERNEL` (`studies/src/statadic_studies/adapters/discrepancy/variants.py:16`); `RBFKernel` (`docs/src/statadic_docs_labs/native.py:25`); `rbf_kernel` (`studies/src/statadic_studies/adapters/inference.py:12`).
- `statadic.geometry.median`: `spatial_medians` (`studies/src/statadic_studies/adapters/population.py:21`); `weiszfeld_medians` (`studies/src/statadic_studies/adapters/population.py:21`).
- `statadic.inference.dispersion`: `dispersion_residual_null` (`studies/tests/test_group_procedures.py:10`).
- `statadic.inference.dyadic`: `DyadicInference` (`studies/src/statadic_studies/adapters/bounds.py:9`); `NodeCountSchedule` (`studies/tests/test_bounds_dyadic.py:7`); `node_bootstrap_report` (`studies/src/statadic_studies/adapters/bounds.py:9`).
- `statadic.inference.equivalence`: `equivalence_test` (`studies/src/statadic_studies/adapters/inference.py:13`).
- `statadic.inference.gel`: `gel_test` (`studies/src/statadic_studies/adapters/estimators.py:26`).
- `statadic.inference.glm_covariance`: `glm_covariance` (`studies/src/statadic_studies/adapters/estimators.py:27`).
- `statadic.inference.gmm`: `gmm_wald` (`studies/src/statadic_studies/adapters/moments.py:13`); `hansen_j` (`studies/src/statadic_studies/adapters/moments.py:13`); `hansen_j_from_fit` (`studies/src/statadic_studies/adapters/estimators.py:28`); `linear_hansen_j_kernel` (`studies/src/statadic_studies/adapters/moments.py:13`).
- `statadic.inference.gmm.covariance`: `gmm_covariance` (`studies/src/statadic_studies/adapters/estimators.py:29`).
- `statadic.inference.gms`: `ProjectionUnavailableError` (`studies/src/statadic_studies/adapters/bounds.py:10`); `indicator_gms` (`studies/src/statadic_studies/adapters/bounds.py:10`).
- `statadic.inference.iv`: `iv_covariance` (`studies/src/statadic_studies/adapters/extended.py:144`).
- `statadic.inference.kappa`: `boundary_robust_pvalue` (`studies/src/statadic_studies/adapters/bounds.py:11`); `kappa_delta_method` (`studies/src/statadic_studies/adapters/kappa.py:6`); `moment_jacobian_concentration` (`studies/src/statadic_studies/adapters/kappa.py:6`).
- `statadic.inference.likelihood`: `likelihood_covariance` (`studies/src/statadic_studies/adapters/extended.py:28`).
- `statadic.inference.linear_covariance`: `classical_covariance` (`performance/src/statadic_performance/workloads.py:14`); `cluster_covariance` (`studies/src/statadic_studies/adapters/dependence/covariance.py:10`); `hac_covariance` (`studies/src/statadic_studies/adapters/dependence/covariance.py:10`); `hc_covariance` (`studies/src/statadic_studies/adapters/dependence/linear_panel.py:17`).
- `statadic.inference.mantel`: `mantel_bootstrap_ci` (`studies/src/statadic_studies/adapters/resampling.py:17`); `mantel_bootstrap_kernel` (`studies/src/statadic_studies/adapters/resampling.py:17`); `paired_mantel_bootstrap` (`studies/src/statadic_studies/adapters/resampling.py:17`).
- `statadic.inference.mixture`: `corrected_mixture_test` (`performance/history/benchmark_jit.py:36`).
- `statadic.inference.multiplicity`: `equivalence_multiplicity_control` (`studies/src/statadic_studies/adapters/inference.py:14`).
- `statadic.inference.panel_covariance`: `fama_macbeth_covariance` (`studies/src/statadic_studies/adapters/dependence/covariance.py:15`); `panel_covariance` (`studies/src/statadic_studies/adapters/dependence/covariance.py:15`).
- `statadic.inference.parameters`: `coefficient_intervals` (`studies/src/statadic_studies/adapters/dependence/_common.py:5`); `linear_restriction_test` (`studies/src/statadic_studies/adapters/dependence/_common.py:5`).
- `statadic.inference.resampling.block_length`: `heuristic_block_length_result` (`studies/src/statadic_studies/adapters/inference.py:15`); `optimal_block_length_result` (`studies/src/statadic_studies/adapters/inference.py:15`); `stationary_bootstrap_estimate` (`studies/src/statadic_studies/adapters/inference.py:15`).
- `statadic.inference.resampling.bootstrap`: `bootstrap_confidence_interval` (`performance/history/benchmark_jit.py:39`); `bootstrap_estimate` (`studies/src/statadic_studies/adapters/inference.py:20`); `bootstrap_mixture_energy_interval` (`studies/src/statadic_studies/adapters/discrepancy/mixture_bootstrap.py:7`).
- `statadic.inference.resampling.permutation`: `energy_permutation_test_result` (`docs/src/statadic_docs_labs/native.py:26`); `permutation_test_result` (`performance/history/benchmark_jit.py:42`).
- `statadic.inference.resampling.permutation.test`: `permutation_test_result` (`studies/src/statadic_studies/adapters/inference.py:21`).
- `statadic.inference.resampling.results`: `HypothesisTest` (`performance/history/benchmark_jit.py:48`); `to_host_test_result` (`performance/history/benchmark_jit.py:43`).
- `statadic.inference.resampling.schedule`: `ReplicateResult` (`studies/src/statadic_studies/adapters/resampling.py:24`); `linear_pairs_replicates` (`performance/src/statadic_performance/workloads.py:15`); `percentile_summary` (`studies/src/statadic_studies/adapters/resampling.py:24`); `refit_replicates` (`studies/src/statadic_studies/adapters/resampling.py:24`); `row_resample_schedule` (`performance/src/statadic_performance/workloads.py:15`).
- `statadic.inference.sharpe`: `deflated_sharpe_ratio` (`studies/src/statadic_studies/adapters/sharpe.py:8`); `multiple_sharpe_test_kernel` (`studies/src/statadic_studies/adapters/sharpe.py:8`); `sharpe_difference_test_kernel` (`studies/src/statadic_studies/adapters/sharpe.py:8`); `sharpe_ratio_ci` (`studies/src/statadic_studies/adapters/sharpe.py:8`).
- `statadic.inference.spatial`: `spatial_ak_test` (`studies/src/statadic_studies/adapters/dependence/spatial_diagnostics.py:15`); `spatial_common_factor_test` (`studies/src/statadic_studies/adapters/dependence/spatial_diagnostics.py:15`); `spatial_covariance` (`studies/src/statadic_studies/adapters/dependence/spatial.py:10`); `spatial_diagnostics` (`studies/src/statadic_studies/adapters/dependence/spatial_diagnostics.py:15`); `spatial_impacts` (`studies/src/statadic_studies/adapters/dependence/spatial_impacts.py:15`); `spatial_likelihood_ratio` (`studies/src/statadic_studies/adapters/dependence/spatial_diagnostics.py:15`); `spatial_panel_impacts` (`studies/src/statadic_studies/adapters/dependence/spatial_impacts.py:15`).
- `statadic.inference.studentized`: `studentized_mixture_statistic` (`studies/src/statadic_studies/adapters/resampling.py:31`).
- `statadic.inference.survival`: `observed_information` (`studies/src/statadic_studies/adapters/estimators.py:30`).
- `statadic.inference.wolak`: `chi_bar_squared_weights_mc_kernel` (`studies/src/statadic_studies/adapters/chibar.py:8`); `wolak_statistic` (`studies/tests/test_bounds_dyadic.py:13`); `wolak_test_kernel` (`studies/src/statadic_studies/adapters/bounds.py:12`).
- `statadic.model.dyadic`: `DyadicInputError` (`studies/src/statadic_studies/adapters/bounds.py:13`); `DyadicModel` (`studies/src/statadic_studies/adapters/bounds.py:13`); `DyadicSample` (`studies/src/statadic_studies/adapters/bounds.py:13`).
- `statadic.model.survival`: `LaggedLogTime` (`studies/tests/dependence/test_spatial_survival_pilot.py:11`); `LogLogisticAFT` (`studies/src/statadic_studies/adapters/extended.py:77`); `LogNormalAFT` (`studies/src/statadic_studies/adapters/estimators.py:31`); `WeibullAFT` (`studies/src/statadic_studies/adapters/estimators.py:31`); `survival_mean` (`studies/src/statadic_studies/adapters/extended.py:77`).
- `statadic.operators.covariance`: `sigma_dist_hetero_kappa` (`studies/src/statadic_studies/adapters/kappa.py:7`).
- `statadic.operators.longrun`: `hall_centered_hac` (`studies/src/statadic_studies/adapters/longrun.py:7`); `newey_west` (`studies/src/statadic_studies/adapters/longrun.py:7`); `optimal_bandwidth` (`studies/src/statadic_studies/adapters/moments.py:20`).
- `statadic.operators.panel`: `prepare_panel_index` (`studies/src/statadic_studies/adapters/dependence/covariance.py:16`).
- `statadic.operators.spatial`: `ObservedSpatialPeriod` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:22`); `prepare_spatial_panel_weights` (`studies/src/statadic_studies/adapters/dependence/spatial_extensions.py:22`); `prepare_spatial_weights` (`studies/src/statadic_studies/adapters/dependence/spatial.py:11`).
- `statadic.optimize.integration`: `GaussianIntegration` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:17`); `IntegrationDesign` (`studies/src/statadic_studies/adapters/dependence/spatial_survival.py:17`).
- `statadic.optimize.psd`: `ridge_psd` (`studies/src/statadic_studies/adapters/covariance_tuning.py:15`).
- `statadic.sample.alternatives`: `AltSpec` (`studies/src/statadic_studies/adapters/population.py:22`); `draw` (`studies/src/statadic_studies/adapters/population.py:22`).
- `statadic.sample.packed`: `PackedClouds` (`docs/src/statadic_docs_labs/native.py:27`); `masked_sum` (`docs/tests/test_implementation_correspondence.py:8`); `pack` (`studies/src/statadic_studies/adapters/discrepancy/fixed_mixture.py:9`); `pair_chunks` (`docs/tests/test_implementation_correspondence.py:8`).
- `statadic.testing.qualification`: `TrialOutcome` (`studies/src/statadic_studies/analysis.py:12`); `summarize_trials` (`studies/src/statadic_studies/analysis.py:12`); `trial_key` (`studies/tests/test_campaign.py:10`).

Child modules:

- `statadic.estimation.iv`: `studies/tests/test_extended.py:104`.
- `statadic.geometry.ground.metrics`: `performance/history/benchmark_jit.py:35`.

Plain module imports:

- `statadic` (`performance/src/statadic_performance/worker.py:47`).
- `statadic` (`studies/src/statadic_studies/campaign.py:16`).
- `statadic` (`studies/src/statadic_studies/source.py:10`).

### STATADIC examples

Scope `/home/marcussky/Git/statadic/src/python/packages/statadic/examples`: 10 Python files parsed; 22 modules and 45 distinct module/name imports; 0 distinct child modules; 0 plain import statements. Parse errors: 0. Final-inventory missing/private imports: none. Imports absent from HEAD inventory: none.

- `statadic.api.artifacts`: `load_analysis` (`first_regression.py:13`); `load_predictor` (`first_regression.py:13`); `save_analysis` (`first_regression.py:13`).
- `statadic.api.comparison`: `DISCO` (`distribution_workflow.py:12`); `PermutationPlan` (`distribution_workflow.py:12`).
- `statadic.api.data`: `MeasureColumn` (`distribution_workflow.py:13`); `Observations` (`change_covariance.py:4`).
- `statadic.api.design`: `DistanceToLandmarks` (`distribution_workflow.py:14`); `Intercept` (`change_covariance.py:5`); `Linear` (`change_covariance.py:5`).
- `statadic.api.embedding`: `PCoA` (`distribution_workflow.py:15`).
- `statadic.api.inference`: `HC` (`change_covariance.py:6`); `ModelBased` (`gamma_regression.py:9`).
- `statadic.api.panel`: `PanelLayout` (`change_estimator.py:13`); `PanelRegression` (`change_estimator.py:13`); `PanelResult` (`change_estimator.py:17`); `SpatialPanelData` (`spatial_panel_survival.py:8`).
- `statadic.api.plots`: `prediction_line` (`poisson_exposure.py:16`); `residual_plot` (`first_regression.py:16`).
- `statadic.api.regression`: `GammaRegression` (`gamma_regression.py:10`); `IVRegression` (`iv_regression.py:10`); `OLS` (`change_covariance.py:7`); `PoissonRegression` (`poisson_exposure.py:17`).
- `statadic.api.representation`: `Energy` (`change_geometry.py:10`); `MMD` (`distribution_workflow.py:18`); `PairwiseGeometry` (`change_geometry.py:14`).
- `statadic.api.survival`: `RightCensored` (`spatial_panel_survival.py:9`); `SpatialSurvivalInput` (`spatial_panel_survival.py:9`); `SpatialSurvivalRegression` (`spatial_panel_survival.py:9`); `SurvivalRegression` (`spatial_panel_survival.py:9`).
- `statadic.discrepancy.mmd.energy`: `energy_distance` (`energy_and_scale.py:8`).
- `statadic.estimation.iv`: `TwoStageLeastSquares` (`iv_regression.py:11`).
- `statadic.estimation.likelihood`: `FixedDispersion` (`gamma_regression.py:11`); `GammaMLE` (`gamma_regression.py:11`).
- `statadic.estimation.panel`: `Pooled` (`change_estimator.py:14`); `Within` (`change_estimator.py:14`).
- `statadic.estimation.survival`: `SpatialSurvivalML` (`spatial_panel_survival.py:15`); `SurvivalML` (`spatial_panel_survival.py:15`).
- `statadic.geometry.ground.metrics`: `ANGULAR` (`energy_and_scale.py:10`); `EUCLIDEAN` (`energy_and_scale.py:10`).
- `statadic.geometry.ground.similarities`: `RBFKernel` (`distribution_workflow.py:19`).
- `statadic.model.survival`: `LogNormalAFT` (`spatial_panel_survival.py:16`).
- `statadic.operators.spatial`: `prepare_spatial_weights` (`spatial_panel_survival.py:17`).
- `statadic.optimize.integration`: `IntegrationDesign` (`spatial_panel_survival.py:18`).
- `statadic.sample.packed`: `PackedClouds` (`change_geometry.py:11`).

The pricing plain import uses `statadic.__version__` at `apps/pipeline/src/pipeline/stages/papers/paper5/spatial_covariance.py:531`; it is public in both inventories. Studies bounds migrated from private `statadic.inference._common.ProjectionUnavailableError` to the public `statadic.inference.gms.ProjectionUnavailableError` at `studies/src/statadic_studies/adapters/bounds.py:10`, as explicitly allowed. The other app identity absent from HEAD is the newly public `statadic.inference.mantel.mantel_bootstrap_kernel`. No imported pricing or publishing identity was removed or renamed relative to HEAD.

The three app `import statadic` uses are scoped source-identity metadata via `statadic.__file__`: studies `campaign.py:141`, `source.py:18`, and performance `worker.py:181`. Publishing tests and the docs release script also use `statadic.__file__` in generated subprocess strings. These are metadata uses, not supported symbol imports.

Documentation Markdown/RST import-line scan parsed 84 distinct symbols in 39 modules; all are in the final inventory. Two inline prose imports could not be parsed by that line scan; manual inspection identifies public `statadic.estimation.linear.LeastSquares`, `LinearProblem` and `statadic.discrepancy.mmd.energy.energy_distance`. Markdown/RST scan is a supplementary static check and does not execute snippets.

Limits: AST scans do not resolve arbitrary reflection, dynamically built strings, notebook cells, external scripts, or all module-alias attribute chains. Search for `importlib.import_module` and `__import__` found studies installation probing variable `MODULES` at `studies/src/statadic_studies/installation.py:103,118`; it checks installed package identity. Pricing dynamic imports found by that search target other packages. String import examples and non-Python sources are not exhaustively resolved. No heavy tests were run for this read-only audit; checkpoint 1 already records the full `just check` result.

Commands: root `devenv shell -- uv run --project src/python --locked python -` with stdlib AST and JSON, including `git show HEAD:src/lean/registry/public-api.json` inside the pinned shell; `devenv shell -- sh -c` with `rg` for dynamic imports, metadata, and documentation; and focused `sed` inspection. Pricing excluded cached `**/packages/statadic/**` and top-level `src/statadic/**`, as well as build and environment caches.

## Enforcement review corrections

The revised inventory rejects augmented or multiple public declarations and
reports unlisted definitions alongside missing/nonliteral declarations, with
all public-module diagnostics aggregated. The exact runtime `static_jaxtyping`
name is covered by the regression fixture. Registry tests passed 36 tests, Ruff
check and eight-file formatting validation, plus correspondence. Independent
review approved these corrections without rerunning tests.

The stricter integrated scan found only the existing augmented export list in
`statadic.core`; the coordinator returned this to the measures owner and reopened
its declaration acceptance pending the final checkpoint. Consumer identities
are unchanged by consolidating that declaration.

The measures owner consolidated all 71 core exports in identical order, with no
import changes; its independent reviewer approved the correction. Scoped Ruff
checks passed. The coordinator refreshed the inventory manifest; export
identities remain unchanged at 2,104, so the consumer audit remains applicable.
No new consumer migration or owner approval is needed.

## Owner flags and omitted checks

`core.typing.static_jaxtyping` remains public as a reusable typing tool; the
measures record flags that support choice for owner awareness. No name awaits
approval to become private. pricing-perspective was inspected read-only and its
application suites were not run. Arbitrary runtime reflection and notebooks are
not claimed by the static audit. Slow/GPU suites, long qualification campaigns,
release/distribution lanes and hosted CI are outside this naming pass; the
required full repository checks are recorded at the final checkpoint.

## Signature-contract audit follow-up

After the consumer audit, a resolved-target signature audit covered class
generic bounds, string annotations, public fields/methods and re-exported
private implementations. The owners exposed the existing contract types and
comparison provenance closure, with independent review. These additions only
extend the audited export set; every earlier consumer identity is preserved.
The earlier 2,104-identity inventory count above describes the consumer audit's
checkpoint; the final count is recorded at completion. All original consumer
import conclusions remain applicable.

The first final check was interrupted before any corrective source edits, not
counted as passing. The restarted final checkpoint follows source freeze,
inventory refresh, catalogue reconciliation, synchronization and formatting.

## Final acceptance — checkpoint 2

The coordinator established final acceptance from the complete passing
`devenv shell -- just check` after all source edits and cross-review were frozen.
This supersedes the earlier interrupted worker/final runs and includes every
signature-contract correction. Results: registry 36; architecture 13; shapes
seven; fresh STATADIC 2,725 passed / 143 skipped / five warnings; evidence 15;
publishing eight; studies 316 passed / three skipped / three warnings;
performance 39; docs x64-off 86; native x64-on 17 with 69 deselected.
Lean certification and inventory staleness, layout, lint, strict typing, builds,
strict site, native exports, TeX and repository-policy checks all passed.
The final inventory and catalogue contain 2,148 identities. No implementation
source changed after this checkpoint. See the
[full verification record](../../verification.md).
