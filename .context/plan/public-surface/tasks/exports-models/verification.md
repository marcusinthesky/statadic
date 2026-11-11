# Verification

Each candidate was searched with both handoff `rg -n -w` commands across `src/python src/latex` and pricing-perspective excluding cached STATADIC. Consumer imports and result/signature/protocol contracts stay public; app `resampling.py` consumes `mantel_bootstrap_kernel`. Dyadic package re-exports and pricing-perspective imports remain public in their defining modules.

| Module | Listed names | Private names | Reason / evidence |
| --- | --- | --- | --- |
| `statadic.estimation.dyadic.contracts` | `DyadicComputationError`, `DyadicFit` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.estimation.dyadic.fit` | `build_dyadic_design`, `DyadicEstimator` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.estimation.dyadic.solve` | — | `least_squares_host` → `_least_squares_host` | Internal helpers; cross-family uses need scoped Tach interfaces. |
| `statadic.estimation.spatial.observed_panel` | `ObservedSpatialPanelProblem`, `ObservedSpatialPanelFit`, `observed_panel_profile`, `ObservedPanelML` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.clustering` | `cluster_pairs` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.contracts` | `ParameterFit`, `SamplingCovariance`, `FittedUncertainty` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.dyadic.bootstrap` | `draw_node_counts`, `bootstrap_dyadic`, `bootstrap_coefficient_inference`, `node_bootstrap_coefficient` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.dyadic.contracts` | `NodeCountSchedule`, `DyadicInference`, `DyadicFitRequest`, `DyadicBootstrapDesign`, `DyadicBootstrapResult`, `DyadicFitResult` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.gms` | `GmsResult`, `indicator_gms`, `ProjectionUnavailableError` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.sharpe.deflated` | `probabilistic_sharpe_ratio`, `expected_maximum_sharpe_ratio`, `DeflatedSharpeResult`, `deflated_sharpe_ratio` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.sharpe.difference` | `SharpeDifferenceKernelResult`, `sharpe_difference_test_kernel`, `sharpe_difference_test`, `sharpe_ratio_ci` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.sharpe.multiple` | `MultipleSharpeResult`, `MultipleSharpeKernelResult`, `multiple_sharpe_test_kernel`, `multiple_sharpe_test` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.survival.information` | `SurvivalCovariance`, `SurvivalUncertainty`, `observed_information` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.wolak` | `WolakResult`, `ChiBarWeightsKernelResult`, `WolakKernelResult`, `wolak_statistic`, `chi_bar_squared_weights_mc`, `chi_bar_squared_weights_mc_kernel`, `chi_bar_squared_sf_kernel`, `chi_bar_squared_sf`, `wolak_test`, `wolak_test_kernel`, `ProjectionUnavailableError` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.model.dyadic.specification` | `DyadicInputError`, `DyadicSample`, `DyadicModel` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.model.dyadic.validation` | — | `array_shape` → `_array_shape`, `is_finite_real_array` → `_is_finite_real_array`, `validate_sample` → `_validate_sample`, `validate_design_inputs` → `_validate_design_inputs`, `validate_model` → `_validate_model`, `validate_effect_options` → `_validate_effect_options` | Internal helpers; cross-family uses need scoped Tach interfaces. |
| `statadic.model.survival.spatial` | `LaggedLogTime` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.mantel` | `mantel_bootstrap_kernel` | `Float` → `_Float`, `Index` → `_Index` | Mantel aliases appear only in private function annotations; app consumes kernel. |
| `statadic.inference.resampling.permutation.test` | `StatisticFn`, `EnergyStatisticProvenance`, `EnergyLabelPermutation` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.resampling.block_length` | `BlockStatistic`, `EagerBlockStatistic` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.resampling.bootstrap` | `BootstrapStatistic`, `EagerBootstrapStatistic` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.estimation.covariance.shrinkage` | `ShrinkageWeight` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.estimation.likelihood` | `Dispersion` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.estimation.spatial.likelihood` | `SpatialModel` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.panel_covariance` | `PanelEquation` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |
| `statadic.inference.spatial.covariance` | `SpatialEquation` | — | Public scientific operations, exposed contracts or deliberate family re-exports. |

Navigation modules `statadic.estimation`, `statadic.inference`, `statadic.inference.resampling`, and `statadic.model` declare typed empty literal exports. The retired docstring-only `statadic.inference.dyadic.fit` was deleted.

`ProjectionUnavailableError` is re-exported from `statadic.inference.gms` and `statadic.inference.wolak`, both of which call the projection failure path. Studies bounds now imports from `statadic.inference.gms`.

## Coordinator requests

- Replace the studies `_common.ProjectionUnavailableError` exception with a scoped interface for `statadic.inference.gms` and `statadic.inference.wolak`.
- Scope `statadic.estimation.dyadic.solve._least_squares_host` to `statadic.inference.dyadic`; scope `statadic.model.dyadic.validation._array_shape`, `_is_finite_real_array`, `_validate_sample` to `statadic.inference.dyadic`, and `_validate_design_inputs`, `_validate_model` to `statadic.estimation.dyadic`.
- Refresh inventory and reconcile catalogue: newly listed identities and removed inferred imports below; targets follow the existing method mapped to the defining module. Record the private renames in the changelog. No owner flags; no pricing-perspective import was privatized.

Inventory/catalogue additions: `statadic.inference.gms.ProjectionUnavailableError`, `statadic.inference.wolak.ProjectionUnavailableError`, `statadic.inference.mantel.mantel_bootstrap_kernel`, `statadic.inference.resampling.permutation.test.StatisticFn`, `.EnergyStatisticProvenance`, `.EnergyLabelPermutation`, `statadic.inference.resampling.block_length.BlockStatistic`, `.EagerBlockStatistic`, `statadic.inference.resampling.bootstrap.BootstrapStatistic`, `.EagerBootstrapStatistic`, `statadic.estimation.covariance.shrinkage.ShrinkageWeight`, `statadic.estimation.likelihood.Dispersion`, `statadic.estimation.spatial.likelihood.SpatialModel`, `statadic.inference.panel_covariance.PanelEquation`, and `statadic.inference.spatial.covariance.SpatialEquation`.

Catalogue/inventory direct identities removed by explicit leaf declarations (including inferred STATADIC imports and privatized helpers):

- `statadic.estimation.dyadic.contracts.DyadicModel` → `estimation.dyadic.contracts`
- `statadic.estimation.dyadic.contracts.NodeIndex` → `estimation.dyadic.contracts`
- `statadic.estimation.dyadic.fit.ArrayView` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.DyadicComputationError` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.DyadicFit` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.DyadicInputError` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.drop_collinear_columns` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.fwl_coefficient` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.least_squares_host` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.symmetric_dyadic_effects` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.validate_design_inputs` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.fit.validate_model` → `estimation.dyadic.fit`
- `statadic.estimation.dyadic.solve.ArrayLike` → `estimation.dyadic.solve`
- `statadic.estimation.dyadic.solve.DyadicComputationError` → `estimation.dyadic.solve`
- `statadic.estimation.dyadic.solve.least_squares` → `estimation.dyadic.solve`
- `statadic.estimation.dyadic.solve.least_squares_host` → `estimation.dyadic.solve`
- `statadic.estimation.spatial.observed_panel.Array` → `estimation.spatial.observed_panel`
- `statadic.estimation.spatial.observed_panel.ArrayLike` → `estimation.spatial.observed_panel`
- `statadic.estimation.spatial.observed_panel.ObservedSpatialPeriod` → `estimation.spatial.observed_panel`
- `statadic.estimation.spatial.observed_panel.SAR` → `estimation.spatial.observed_panel`
- `statadic.estimation.spatial.observed_panel.SEM` → `estimation.spatial.observed_panel`
- `statadic.estimation.spatial.observed_panel.SolverResult` → `estimation.spatial.observed_panel`
- `statadic.estimation.spatial.observed_panel.validate_parameter_names` → `estimation.spatial.observed_panel`
- `statadic.inference.clustering.Array` → `inference.clustering`
- `statadic.inference.clustering.ArrayLike` → `inference.clustering`
- `statadic.inference.contracts.Array` → `inference.contracts`
- `statadic.inference.contracts.SolverResult` → `inference.contracts`
- `statadic.inference.dyadic.bootstrap.ArrayView` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.DyadicBootstrapDesign` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.DyadicBootstrapResult` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.DyadicComputationError` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.DyadicInputError` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.MIN_ACTIVE_DYADS` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.NodeCountSchedule` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.NodeIndex` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.cell_key` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.drop_collinear_columns` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.least_squares_host` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.validate_bootstrap_design` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.validate_inference` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.validate_node_schedule` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.bootstrap.validate_sample` → `inference.dyadic.bootstrap`
- `statadic.inference.dyadic.contracts.ArrayView` → `inference.dyadic.contracts`
- `statadic.inference.dyadic.contracts.DyadicModel` → `inference.dyadic.contracts`
- `statadic.inference.dyadic.contracts.DyadicSample` → `inference.dyadic.contracts`
- `statadic.inference.dyadic.contracts.NodeIndex` → `inference.dyadic.contracts`
- `statadic.inference.gms.Array` → `inference.gms`
- `statadic.inference.gms.ArrayLike` → `inference.gms`
- `statadic.inference.gms.PROJECTION_TOLERANCE` → `inference.gms`
- `statadic.inference.gms.batched_nonnegative_quadratic_program` → `inference.gms`
- `statadic.inference.gms.cell_key` → `inference.gms`
- `statadic.inference.gms.hall_centered_hac` → `inference.gms`
- `statadic.inference.gms.refine_projection_solution` → `inference.gms`
- `statadic.inference.sharpe.deflated.Array` → `inference.sharpe.deflated`
- `statadic.inference.sharpe.deflated.ArrayLike` → `inference.sharpe.deflated`
- `statadic.inference.sharpe.deflated.sample_skew_kurtosis` → `inference.sharpe.deflated`
- `statadic.inference.sharpe.deflated.sharpe_ratio` → `inference.sharpe.deflated`
- `statadic.inference.sharpe.difference.Array` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.ArrayLike` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.PRNGKey` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.circular_block_bootstrap_indices` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.circular_block_bootstrap_indices_kernel` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.heuristic_block_length` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.heuristic_block_length_result` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.sharpe_ratio_axis` → `inference.sharpe.difference`
- `statadic.inference.sharpe.difference.stationary_bootstrap_ci` → `inference.sharpe.difference`
- `statadic.inference.sharpe.multiple.Array` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.ArrayLike` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.PRNGKey` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.circular_block_bootstrap_indices` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.circular_block_bootstrap_indices_kernel` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.optimal_block_length` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.optimal_block_length_result` → `inference.sharpe.multiple`
- `statadic.inference.sharpe.multiple.sharpe_ratio_axis` → `inference.sharpe.multiple`
- `statadic.inference.survival.information.Array` → `inference.survival.information`
- `statadic.inference.survival.information.InferenceStatus` → `inference.survival.information`
- `statadic.inference.survival.information.SpatialSurvivalFit` → `inference.survival.information`
- `statadic.inference.survival.information.SurvivalFit` → `inference.survival.information`
- `statadic.inference.wolak.Array` → `inference.wolak`
- `statadic.inference.wolak.ArrayLike` → `inference.wolak`
- `statadic.inference.wolak.POLISH_ROUNDS` → `inference.wolak`
- `statadic.inference.wolak.PROJECTION_TOLERANCE` → `inference.wolak`
- `statadic.inference.wolak.batched_nonnegative_quadratic_program` → `inference.wolak`
- `statadic.inference.wolak.cell_key` → `inference.wolak`
- `statadic.inference.wolak.hall_centered_hac` → `inference.wolak`
- `statadic.inference.wolak.hall_centered_hac_kernel` → `inference.wolak`
- `statadic.inference.wolak.refine_projection_solution` → `inference.wolak`
- `statadic.model.dyadic.specification.ArrayView` → `model.dyadic.specification`
- `statadic.model.dyadic.validation.ArrayView` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.DyadicInputError` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.LabelSequence` → `operators.design`
- `statadic.model.dyadic.validation.NodeIndex` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.array_shape` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.is_finite_real_array` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.validate_design_inputs` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.validate_effect_options` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.validate_model` → `model.dyadic.validation`
- `statadic.model.dyadic.validation.validate_sample` → `model.dyadic.validation`

## Cross-review

Reviewed the measure/geometry lane read-only: its diff changes exports only, preserving behavior, signatures, defaults and tests. AST inspection of imports across repository Python and pricing-perspective found no removed symbol imports; navigation namespace imports were submodules. Public result/domain/provenance names and listed definitions were included. Its verification record was not yet present at review time.

## Checks

- `just python::lint`: passed (452 Python files formatted at that checkpoint).
- `just python::architecture`: failed pending scoped Tach interfaces from concurrent lanes and this lane; integration owner handles these.
- `just python::typecheck`: passed after empty-list annotations; strict checks reported zero errors and seven shape tests passed.
- `just python::statadic::test-fresh tests/estimation tests/inference tests/model`: interrupted by coordinator request at 24% (exit 130), no test failures reported before interruption.
- `just python::studies::check`: lint and formatting passed; pytest interrupted by coordinator request at 22% (exit 143), no test failures reported before interruption. Shared suites will run without competing work during coordinator integration.
- `just check` and shared inventory/catalogue refresh: coordinator integration checkpoint; not run by this lane.
- `just python::docs::check`: not run; no documented public name changed.

## Integrated acceptance

The coordinator established acceptance with checkpoint 1: the complete
`devenv shell -- just check` passed after source freeze, scoped Tach integration,
inventory refresh and catalogue reconciliation. This includes 2,725 fresh STATADIC
tests (143 skipped), architecture 13, registry 34, shapes seven, studies 316
(three skipped), docs 86 x64-off and 17 native x64-on (69 deselected). The
concurrent worker crash/timeouts did not recur. See the
[coordination record](../../verification.md).

## Public-signature alias follow-up

A signature audit found seven private aliases used by public signatures or result fields. These were renamed and listed without compatibility aliases: `inference.sharpe.difference.FloatScalar` and `ReturnVector`; `inference.sharpe.multiple.FloatScalar`, `ReturnVector`, `StrategyVector` and `StrategyPanel`; and `estimation.spatial.panel.FixedGraphCache`. Both `_BootstrapMatrix` aliases remain private because they occur only in private helper signatures. The optional `fixed_graph_cache` parameter is public and retains its default and underlying tuple type. Searches of repository Python and LaTeX and filtered pricing-perspective found no external users of the private aliases; pricing-perspective calls `spatial_panel_profile`.

The coordinator must add the seven module-qualified identities to the matching catalogue targets (`inference.sharpe.difference`: two; `inference.sharpe.multiple`: four; `estimation.spatial.panel`: one), refresh inventory and include these clean renames in the changelog. No Tach changes are needed. Scoped Ruff check and formatting passed. A read-only AST scan of public functions and public class fields/methods in the three touched modules found no remaining references to locally defined private classes or type aliases. No focused suite was launched during the coordinator integration checkpoint.

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
