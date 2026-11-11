# API export verification

Both prescribed `rg -n -w NAME` searches were run for each candidate against `src/python src/latex` and pricing-perspective with cached-copy exclusions. No consumer imported a renamed helper. Publishing imports `freeze_metadata` and `metadata_to_json`.

| Module | Names listed | Names made private | Reason |
| --- | --- | --- | --- |
| api | none | none | Navigation namespace. |
| base | RegressionFit | none | Public RegressionResult generic bound. |
| reporting | JsonValue, ReportNumber, freeze_metadata, metadata_to_json, ParameterRow, Metric, ReferenceLine, PlotSpec, Report | finite_metric, render_pgfplots | Publishing uses metadata utilities; types and classes occur in public fields and signatures. finite_metric serves host workflows; render_pgfplots serves PlotSpec and tests. |
| representation | NativeMeasureMatrix, RowId, HilbertianMeasureMatrix, GeometrySpec, MMD, Energy, Wasserstein1D, SlicedWasserstein, PairwiseGeometry, PreparedGeometry | none | Public geometry contracts, inputs, results, protocol and documented workflows. |
| embedding | RowIds, PreparedPCA, PreparedPCoA | none | Public signatures and result context. |
| association | MantelInput, PreparedMantel | none | Public fit and prepare types. |
| covariance | CovarianceContext, PreparedCovariance | none | Public result context and prepare type. |
| data | DtypedColumn, NumericInput | check_array_precision | Public input protocol and alias appear in Observations signatures; precision helper serves inputs and artifacts internally. |
| inference | InferenceContext | none | Protocol for the public InferenceResult context field. |
| inputs | ArrowStream, InputRole, LazyFrame | model_roles | The protocols occur in RegressionInput and InputRole appears in as_observations; role extraction serves base and regression internally. RegressionInput remains public. |
| dyadic | PreparedDyadic | none | Public prepare and estimate_prepared type. |
| plots | KindPredictor, Predictor | none | Predictor protocols for public prediction_line and its kind option. |
| tables | ArrowFrame | none | User protocol in public conversion signatures. |

Existing exports remain listed. No new re-exports were needed; types are declared in their public defining modules. Existing public package re-exports remain intact. `_confidence_label` keeps its existing Tach exception.

## Coordinator requests

- Add scoped Tach interfaces: `statadic.api.reporting._finite_metric` to `statadic.api.panel`, `statadic.api.regression`, `statadic.api.spatial`, `statadic.api.survival`; `statadic.api.data._check_array_precision` to `statadic.api.inputs`, `statadic.api.artifacts`; `statadic.api.inputs._model_roles` to `statadic.api.base`, `statadic.api.regression`.
- Read-only inventory comparison found additions: `statadic.api.association.{MantelInput,PreparedMantel}`, `statadic.api.covariance.{CovarianceContext,PreparedCovariance}`, `statadic.api.data.NumericInput`, `statadic.api.dyadic.PreparedDyadic`, `statadic.api.embedding.{PreparedPCA,PreparedPCoA,RowIds}`, `statadic.api.inputs.{ArrowStream,InputRole,LazyFrame}`, `statadic.api.reporting.{JsonValue,ReportNumber}`, `statadic.api.representation.{NativeMeasureMatrix,RowId}`, `statadic.api.tables.ArrowFrame` (17). Map each to its existing module method owner in the catalogue. The two protocols were added after the comparison in response to cross-review.
- Removed inventory entries: `statadic.api.reporting.{finite_metric,render_pgfplots}` and the formerly inferred `statadic.api.representation.{Array,ConstructionOrigin,DMat,EmpiricalDistribution,GroundDistanceSelection,HilbertianPseudometricLaw,MatrixProvenance,PackedClouds,PositiveSemidefiniteKernelLaw,ProjectionDesign,SimilarityStrategy,checked_packed_clouds,finite_sliced_w2_matrix,measure_reference,quantile_w2_matrix,resolve_energy_geometry,transport_cross,weighted_energy_cross,weighted_energy_matrix,weighted_mmd_cross,weighted_mmd_matrix}` (23). Remove corresponding catalogue entries. These representation imports were incidental bindings; their defining modules retain their own declarations.
- Changelog: clean-break internal renames `finite_metric` to `_finite_metric`, `render_pgfplots` to `_render_pgfplots`, `check_array_precision` to `_check_array_precision`, `model_roles` to `_model_roles`; expose `_ArrowStream` and `_LazyFrame` as `ArrowStream` and `LazyFrame`; formerly inferred representation imports no longer declared exports. No behavior or numeric change.
- Owner flags: none.

Signature-audit follow-up: add `statadic.api.base.RegressionFit`, `statadic.api.data.DtypedColumn`, `statadic.api.inference.InferenceContext`, `statadic.api.plots.{KindPredictor,Predictor}`, `statadic.api.spatial.{InstrumentRoles,SLXColumns,SpatialCoreFit,SpatialEstimator,PreparedObservedSpatialPanel}`, `statadic.api.regression.PreparedIV`, `statadic.api.comparison.{ComparisonCore,GroupLabels,NativeDecomposition,RowIds}` and their inventoried members to the matching catalogue method mappings. Changelog clean private-to-public renames `_RegressionFit`, `_DtypedColumn`, `_InferenceContext`, `_KindPredictor`, `_Predictor`, `_InstrumentRoles`, `_SLXColumns` to those names. No new Tach interfaces. The spatial, regression and comparison re-exports expose types defined in private implementation modules without changing the underlying definitions. An AST audit of public class fields, methods, bases, generic bounds and string annotations identified these types; an internal artifact storage `Arrays` class is not a public contract. The coordinator will perform a global signature audit after source freeze.

The coordinator's global resolved-target audit additionally identified `ComparisonTest`, used in `ComparisonInference.calibrated`; it is now re-exported from `statadic.api.comparison`. Add its inventory and catalogue entries and note the export in the changelog. Scoped Ruff check and format passed; focused tests were not repeated for this declaration-only correction.

The return-contract closure of `ComparisonTest` also re-exports `ComparisonStatistic`, `RandomCalibration`, `EnumerationCalibration`, `ResidualCalibration`, `SquaredDistanceAnovaOrigin`, `SpatialMedianDispersionOrigin`, `RandomOuterLabelPermutation`, `ExactOuterLabelPermutation`, and `RandomResidualPermutation`. These nine existing definitions originate in `_shared`; add their inventory and catalogue entries and changelog export listing. Scoped Ruff check and format passed. The private implementation base `_GroupComparison` is not a user signature contract and remains private.

## Checks

All commands ran from repository root in `devenv shell --`. Initial sandboxed devenv failed to connect to the Nix daemon; the approved invocation succeeded. First owned-file Ruff check fixed three import issues and reported five line-length issues; explicit Ruff formatting and owned-file Ruff check passed (5 files reformatted). `just python::lint` passed (453 files formatted). `just python::architecture` failed on 14 imports pending the three Tach interfaces above. First `just python::typecheck` found the empty API `__all__` lacked an annotation; fixed. Second found six unannotated empty namespace lists in the concurrent measures lane, reported to that worker. Focused fresh tests and final counts follow. `just python::docs::check` was not run: no documented name changed. `just check` and shared refreshes are reserved for the coordinator's integrated checkpoint.

After cross-review, `just python::lint` passed again (452 files formatted); `just python::statadic::test-fresh tests/api/test_inputs.py` passed 12 tests. The third `just python::typecheck` reported zero static errors, but its shape lane had 6 passes and 1 subprocess timeout after 30 seconds under concurrent load. The broad `just python::statadic::test-fresh tests/api tests/integration/api` reported 291 passed, 3 skipped, and 1 failed after its worker crashed on `test_shared_geometry_supports_group_analysis_and_prediction[comparison1]`; it was interrupted during shutdown (exit 130) to leave resources for the integrated checkpoint. A serial integrated rerun is needed to settle both outcomes.

Signature-audit follow-up: both prescribed searches found the five private protocol names only in their defining modules. `devenv shell -- sh -c 'cd src/python && uv run --locked ruff format packages/statadic/src/statadic/api/base.py packages/statadic/src/statadic/api/data.py packages/statadic/src/statadic/api/inference.py packages/statadic/src/statadic/api/plots.py && uv run --locked ruff check packages/statadic/src/statadic/api/base.py packages/statadic/src/statadic/api/data.py packages/statadic/src/statadic/api/inference.py packages/statadic/src/statadic/api/plots.py'` passed (one file formatted). `devenv shell -- just python::statadic::test-fresh tests/api/test_inputs.py tests/api/test_plots.py tests/api/test_inference.py tests/api/test_base_lifecycle.py` passed 28 tests. Full checks and inventory refresh remain with the coordinator.

Scoped Ruff check and format for the additional spatial, regression and comparison files passed; an initial check on comparison flagged export ordering, corrected and rechecked. An attempted `just python::statadic::test-fresh tests/api/test_spatial.py tests/api/test_comparison.py tests/api/test_iv.py` collected no tests (exit 5: wrong IV path); the corrected command using `tests/api/regression/test__iv.py` passed 48 tests with 2 skipped. The measures worker cross-reviewed and approved the signature corrections and subsequent comparison re-exports.

## Cross-review of exports-models

Read the task specification and model, inference, estimation and mirrored test diffs; checked renamed dyadic helper consumers and projection exception call sites. Found that public Wolak calls also reach `ProjectionUnavailableError` and requested its re-export from `inference.wolak`; also flagged an unannotated empty `__all__` in dyadic validation. Confirmed both corrections. The reviewed diff otherwise consists of exports, requested retired-module removal and consistent helper renames, with tests edited to follow renames.

Reviewed the later seven model-lane signature alias corrections in `estimation/spatial/panel.py` and `inference/sharpe/{difference,multiple}.py`: all are listed at their defining public modules and consistently renamed; underlying types and the cache parameter default are unchanged. No findings.

## Integrated acceptance

The coordinator established acceptance with checkpoint 1: the complete
`devenv shell -- just check` passed after source freeze, scoped Tach integration,
inventory refresh and catalogue reconciliation. This includes 2,725 fresh STATADIC
tests (143 skipped), architecture 13, registry 34, shapes seven, studies 316
(three skipped), docs 86 x64-off and 17 native x64-on (69 deselected). The
concurrent worker crash/timeouts did not recur. See the
[coordination record](../../verification.md).

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
