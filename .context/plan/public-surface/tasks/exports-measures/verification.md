# Measures export verification

Both required `rg -n -w NAME` consumer searches were run for every candidate across `src/python src/latex` and pricing-perspective, excluding cached STATADIC copies. Inventory and catalogue occurrences were not treated as consumers. No definition was renamed or made private.

| Module | Names listed | Private | Reason and consumer evidence |
| --- | --- | --- | --- |
| association, discrepancy, geometry, geometry.barycentre, operators, sample | empty literal lists | none | Navigation namespaces, without deliberate re-exports. |
| association.decomposition | DiscoIndexProvenance, DiscoFRatioProvenance | none | Public `DiscoDecomposition` result fields. |
| core.domains | NormalizationPullbackOrigin, UnitNormSupport | none | Public transform and ground-geometry contract types; sample.evidence documents UnitNormSupport. |
| core.typing | static_jaxtyping | none | Cross-module typing tool; flagged below. |
| sample.alternatives | AltSpec, draw | none | Studies population adapter imports both. |
| sample.packed | PackedClouds, checked_packed_clouds, pack, masked_row_mask, masked_sum, masked_mean, pair_chunks | none | pricing pipeline imports PackedClouds; docs method catalogue names reductions and pair_chunks; public factory. |
| sample.reduction | Projection, fit_projection | none | pricing imports Projection; documented fit_projection and public return type. |
| optimize.integration.gaussian | GaussianIntegration, GaussianIntegrationResult, IntegrationDesign, InverseNormalResult, gaussian_log_survival, inverse_log_normal_result | none | All deliberately re-exported by optimize.integration. |
| geometry.barycentre.mmd.solve | EnergyBarycentreDiagnostics, KernelBarycentreDiagnostics | none | Public result diagnostics; latter deliberately re-exported by family. |
| discrepancy.transport.matrix | FiniteSlicedW2Origin, QuantileW2Origin, finite_sliced_w2_matrix, quantile_w2_matrix, transport_cross | none | Public matrix provenance and operations; studies adapter imports transport_cross. |
| discrepancy.mmd.weighted | measure_reference, weighted_mmd_cross, weighted_mmd_matrix, weighted_mmd_squared_cross | none | Public weighted comparison operations; studies adapter uses weighted_mmd_cross; api representation uses measure_reference. |
| discrepancy.mmd.energy.weighted | weighted_energy_cross, weighted_energy_matrix | none | Studies adapter and tests import the operations. |
| discrepancy.mmd.energy.balancing.effective_size | inverse_variance_m_eff | none | Family re-export and docs package-development. |
| discrepancy.mmd.energy.balancing.mixture | energy_distance_kernel_mixture | none | Family re-export and studies fixed-mixture adapter. |
| discrepancy.mmd.energy.balancing.statistics | energy_test_kernel, energy_test_kernel_pairwise_max | none | Family re-exports and notebook consumer. |
| discrepancy.mmd.energy.balancing.weights | energy_distance_kernel_pairwise, kernel_balancing_weights | none | Family re-exports; studies resampling adapter imports kernel_balancing_weights. |

No new rule-2 re-exports are necessary: contract types are public at their defining modules; existing family re-exports remain in place. No Tach interfaces or CHANGELOG rename entries are requested. Existing scoped core interfaces are unchanged.

Owner flag: `core.typing.static_jaxtyping` was retained as a reusable cross-module tool rather than renamed across many modules. It was previously unlisted; confirm whether external support is desired.

## Cross-review

Worker C approved the measures declaration-only source diff. API lane review found that `RegressionInput` included private user-implemented `_ArrowStream` and `_LazyFrame` protocols. Worker A exposed them as `ArrowStream` and `LazyFrame`. Follow-up review approved the correction: signatures and defaults remain unchanged, and test diffs follow renames without assertion changes.

Enforcement follow-up: consolidated the two `core.__all__` literals into one, preserving import order. An AST comparison against HEAD found 71 exports before and after, with the same sequence and set. Scoped locked Ruff format left the file unchanged and Ruff check passed. The `RUF022` suppression preserves the existing export order. Coordinator will run integrated checks and refresh the inventory.

Signature audit follow-up: renamed private aliases `_UStatisticEstimates`, `_VStatisticEstimates`, and `_RawIdentityTransform` to `UStatisticEstimates`, `VStatisticEstimates`, and `RawIdentityTransform`, and listed them in their defining modules. Both required consumer searches found only local references. Their constituent contract types are all listed by their public defining modules. Scoped locked Ruff format left both files unchanged and Ruff check passed; no heavy checks were run. The coordinator must add three inventory and catalogue identities in addition to the five previously listed below:

Worker C approved these three alias corrections. Read-only review of A's signature corrections approved their protocol renames and public re-exports; protocol signatures, defaults and runtime-checkable decorators remain intact.

- `statadic.discrepancy.mmd.branded.UStatisticEstimates` → same target (mmd).
- `statadic.discrepancy.mmd.branded.VStatisticEstimates` → same target (mmd).
- `statadic.geometry.ground.similarities.RawIdentityTransform` → same target (ground).

The private aliases were not public inventory identities, so no public inventory removals or CHANGELOG renames arise from this follow-up. Constant-only `_NormalizationTransform` remains outside this correction.

Resolved signature audit follow-up: re-exported `SimilarityFunction` alongside `declare_similarity` at both `statadic.geometry.ground` and `statadic.geometry.ground.similarities`; the existing definition is unchanged. Both required consumer searches found only its defining module. Scoped Ruff format left two files unchanged and Ruff check passed. Add both public identities to inventory and catalogue; each resolves to `statadic.geometry.ground._similarity_strategy.SimilarityFunction` (method `ground`). No Tach or CHANGELOG rename is needed. API cross-review approved the final `ComparisonTest` re-export for `ComparisonInference.calibrated`.

Worker C reviewed and approved both `SimilarityFunction` re-exports.

## Commands

All commands used `devenv shell --` at the root and the locked Python workspace. An initial sandboxed shell attempt failed to connect to the Nix daemon; the escalated rerun succeeded.

- Consumer searches: both required searches for all 41 candidates; additional read-only import search covered formerly implicit exports.
- Ruff format and lint fixes ran only on owned changed files: 15 formatted, six unchanged; three ordering issues fixed. `git diff --check` passed.
- `just python::lint`: initial run found three owned ordering issues (fixed); rerun found three transient API protocol lint issues, reported to A.
- `just python::architecture`: failed with 21 other-lane pending Tach interface violations; coordinator integration owns these.
- `just python::typecheck`: initial run found 12 untyped empty lists (six owned and fixed); rerun found one other-lane empty list in model/dyadic/validation.py.
- `just python::statadic::test-fresh tests/core tests/sample tests/optimize tests/geometry tests/discrepancy tests/association tests/operators`: interrupted at 76% (exit 130), with no failures shown, to end competing numerical suites; no final test count is available. Coordinator will rerun.
- `just python::docs::check`: not run; no documented name was renamed.
- `just check`: coordinator runs at integrated checkpoint, after the shared inventory refresh and concurrent lanes.

## Coordinator inventory and catalogue requests

The following read-only comparison is against the checked-in registry. Each line gives identity, resolved target, and inventory method ID. Refresh using `just lean::inventory-refresh` and reconcile catalogue identities. Removed identities are former implicit imports; no external import of those identities was found in the consumer import search.

### Added (5)

- `statadic.association.decomposition.DiscoFRatioProvenance` → `statadic.association.decomposition.DiscoFRatioProvenance` (association)
- `statadic.association.decomposition.DiscoIndexProvenance` → `statadic.association.decomposition.DiscoIndexProvenance` (association)
- `statadic.core.domains.NormalizationPullbackOrigin` → `statadic.core.domains.NormalizationPullbackOrigin` (core)
- `statadic.core.domains.UnitNormSupport` → `statadic.core.domains.UnitNormSupport` (core)
- `statadic.core.typing.static_jaxtyping` → `statadic.core.typing.static_jaxtyping` (core)

### Removed (64)

- `statadic.discrepancy.mmd.energy.balancing.mixture.GroundDistanceSelection` → `statadic.geometry.ground.config.GroundDistanceSelection` (energy)
- `statadic.discrepancy.mmd.energy.balancing.mixture.PackedClouds` → `statadic.sample.packed.PackedClouds` (energy)
- `statadic.discrepancy.mmd.energy.balancing.mixture.candidate_energy_components` → `statadic.discrepancy.mmd.components.candidate_energy_components` (energy)
- `statadic.discrepancy.mmd.energy.balancing.mixture.cdist` → `statadic.geometry.ground.metrics.cdist` (energy)
- `statadic.discrepancy.mmd.energy.balancing.mixture.resolve_ground_distance` → `statadic.geometry.ground.config.resolve_ground_distance` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.GroundDistanceSelection` → `statadic.geometry.ground.config.GroundDistanceSelection` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.PackedClouds` → `statadic.sample.packed.PackedClouds` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.candidate_energy_components` → `statadic.discrepancy.mmd.components.candidate_energy_components` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.cdist` → `statadic.geometry.ground.metrics.cdist` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.inverse_variance_m_eff` → `statadic.discrepancy.mmd.energy.balancing.effective_size.inverse_variance_m_eff` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.pack` → `statadic.sample.packed.pack` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.pgd_simplex_affine` → `statadic.optimize.simplex.pgd_simplex_affine` (energy)
- `statadic.discrepancy.mmd.energy.balancing.statistics.resolve_ground_distance` → `statadic.geometry.ground.config.resolve_ground_distance` (energy)
- `statadic.discrepancy.mmd.energy.balancing.weights.GroundDistanceSelection` → `statadic.geometry.ground.config.GroundDistanceSelection` (energy)
- `statadic.discrepancy.mmd.energy.balancing.weights.PackedClouds` → `statadic.sample.packed.PackedClouds` (energy)
- `statadic.discrepancy.mmd.energy.balancing.weights.candidate_energy_components` → `statadic.discrepancy.mmd.components.candidate_energy_components` (energy)
- `statadic.discrepancy.mmd.energy.balancing.weights.pack` → `statadic.sample.packed.pack` (energy)
- `statadic.discrepancy.mmd.energy.balancing.weights.pgd_simplex_affine` → `statadic.optimize.simplex.pgd_simplex_affine` (energy)
- `statadic.discrepancy.mmd.energy.balancing.weights.resolve_ground_distance` → `statadic.geometry.ground.config.resolve_ground_distance` (energy)
- `statadic.discrepancy.mmd.energy.weighted.Array` → `jaxtyping.Array` (energy)
- `statadic.discrepancy.mmd.energy.weighted.DMat` → `statadic.core.matrices.DMat` (energy)
- `statadic.discrepancy.mmd.energy.weighted.EmpiricalDistribution` → `statadic.core.domains.EmpiricalDistribution` (energy)
- `statadic.discrepancy.mmd.energy.weighted.GroundDistance` → `statadic.geometry.ground._strategy.GroundDistance` (energy)
- `statadic.discrepancy.mmd.energy.weighted.GroundDistanceSelection` → `statadic.geometry.ground.config.GroundDistanceSelection` (energy)
- `statadic.discrepancy.mmd.energy.weighted.HilbertianPseudometricLaw` → `statadic.core.matrices.HilbertianPseudometricLaw` (energy)
- `statadic.discrepancy.mmd.energy.weighted.Law` → `statadic.core.axioms.Law` (energy)
- `statadic.discrepancy.mmd.energy.weighted.MatrixProvenance` → `statadic.core.matrices.MatrixProvenance` (energy)
- `statadic.discrepancy.mmd.energy.weighted.PackedClouds` → `statadic.sample.packed.PackedClouds` (energy)
- `statadic.discrepancy.mmd.energy.weighted.RootEnergyMetric` → `statadic.core.matrices.RootEnergyMetric` (energy)
- `statadic.discrepancy.mmd.energy.weighted.VStatisticScheme` → `statadic.core.matrices.VStatisticScheme` (energy)
- `statadic.discrepancy.mmd.energy.weighted.cdist` → `statadic.geometry.ground.metrics.cdist` (energy)
- `statadic.discrepancy.mmd.energy.weighted.resolve_energy_geometry` → `statadic.discrepancy.mmd.energy.geometry.resolve_energy_geometry` (energy)
- `statadic.discrepancy.mmd.weighted.Array` → `jaxtyping.Array` (mmd)
- `statadic.discrepancy.mmd.weighted.DMat` → `statadic.core.matrices.DMat` (mmd)
- `statadic.discrepancy.mmd.weighted.EmpiricalDistribution` → `statadic.core.domains.EmpiricalDistribution` (mmd)
- `statadic.discrepancy.mmd.weighted.HilbertianPseudometricLaw` → `statadic.core.matrices.HilbertianPseudometricLaw` (mmd)
- `statadic.discrepancy.mmd.weighted.LINEAR_KERNEL` → `statadic.geometry.ground.similarities.LINEAR_KERNEL` (mmd)
- `statadic.discrepancy.mmd.weighted.MatrixProvenance` → `statadic.core.matrices.MatrixProvenance` (mmd)
- `statadic.discrepancy.mmd.weighted.PackedClouds` → `statadic.sample.packed.PackedClouds` (mmd)
- `statadic.discrepancy.mmd.weighted.PositiveSemidefiniteKernelLaw` → `statadic.geometry.ground._similarity_strategy.PositiveSemidefiniteKernelLaw` (mmd)
- `statadic.discrepancy.mmd.weighted.RBFKernel` → `statadic.geometry.ground.similarities.RBFKernel` (mmd)
- `statadic.discrepancy.mmd.weighted.RootedMmd` → `statadic.discrepancy.provenance.RootedMmd` (mmd)
- `statadic.discrepancy.mmd.weighted.SimilarityStrategy` → `statadic.geometry.ground._similarity_strategy.SimilarityStrategy` (mmd)
- `statadic.discrepancy.mmd.weighted.VStatisticScheme` → `statadic.core.matrices.VStatisticScheme` (mmd)
- `statadic.discrepancy.mmd.weighted.gram` → `statadic.geometry.ground.similarities.gram` (mmd)
- `statadic.discrepancy.transport.matrix.Array` → `jaxtyping.Array` (transport)
- `statadic.discrepancy.transport.matrix.DMat` → `statadic.core.matrices.DMat` (transport)
- `statadic.discrepancy.transport.matrix.EmpiricalDistribution` → `statadic.core.domains.EmpiricalDistribution` (transport)
- `statadic.discrepancy.transport.matrix.HilbertianPseudometricLaw` → `statadic.core.matrices.HilbertianPseudometricLaw` (transport)
- `statadic.discrepancy.transport.matrix.MatrixProvenance` → `statadic.core.matrices.MatrixProvenance` (transport)
- `statadic.discrepancy.transport.matrix.PackedClouds` → `statadic.sample.packed.PackedClouds` (transport)
- `statadic.discrepancy.transport.matrix.ProjectionDesign` → `statadic.discrepancy.transport.sliced.ProjectionDesign` (transport)
- `statadic.discrepancy.transport.matrix.TransportOrigin` → `statadic.core.domains.TransportOrigin` (transport)
- `statadic.discrepancy.transport.matrix.sliced_wasserstein_under_design` → `statadic.discrepancy.transport.sliced.sliced_wasserstein_under_design` (transport)
- `statadic.discrepancy.transport.matrix.wasserstein_1d` → `statadic.discrepancy.transport.univariate.wasserstein_1d` (transport)
- `statadic.sample.alternatives.Dimension` → `statadic.core.typing.Dimension` (sample)
- `statadic.sample.alternatives.DimensionVar` → `statadic.core.typing.DimensionVar` (sample)
- `statadic.sample.alternatives.ShapedArray` → `statadic.core.typing.ShapedArray` (sample)
- `statadic.sample.packed.Array` → `jaxtyping.Array` (sample)
- `statadic.sample.packed.ArrayLike` → `jaxtyping.ArrayLike` (sample)
- `statadic.sample.reduction.DimensionVar` → `statadic.core.typing.DimensionVar` (sample)
- `statadic.sample.reduction.Elements` → `statadic.core.typing.Elements` (sample)
- `statadic.sample.reduction.Shape` → `statadic.core.typing.Shape` (sample)
- `statadic.sample.reduction.ShapedArray` → `statadic.core.typing.ShapedArray` (sample)

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
