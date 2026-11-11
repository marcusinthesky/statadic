# Dead-code audit verification

The final locked `just python::statadic::dead-code` command reported 142 reviewed
candidates (exit 3) after integration and formatting. A finding is not proof that
a symbol is unused. The retained [raw report](tasks/dead-code-cleanup/audit.txt)
contains the complete list; every retained category is accounted for below. No
blanket whitelist or suppression hides these advisory findings.

| Candidate | Disposition and evidence |
| --- | --- |
| `components._compute_energy_components_batched`, `_compute_energy_components_packed`, `_mean_ground_distance_matrix_packed`, `_triu_pair_chunks` | Removed obsolete private chain; its test-only callers now exercise active public mean-distance paths or the existing `pair_chunks` contract. |
| `energy.matrix._shared_cloud_shape`, `_require_stackable_shape`, `_normalized_clouds`, `_energy_distance_matrix_core`, `_energy_distance_matrix_packed_core`, `_mean_distance_matrix_core`; `mmd.matrix._require_nonempty_clouds` | Removed after source caller review. Poisoned-padding and traceability witnesses exercise public `energy_distance_matrix`. |
| `tests/estimation/survival/test_fit.py::rv_continuous_frozen`, `tests/model/survival/test_aft.py::rv_continuous_frozen` | Retained: used in quoted `cast` annotations. |
| `optimize.psd._positive_projection_jvp`, `_positive_energy_jvp` | Retained: registered through the corresponding `defjvp` decorators. |
| `core.typing.__class_getitem__`, `operators.design.__getitem__` | Retained: Python special methods invoked by language operations. |
| `tests/conftest.py::pytest_plugins`, `pytest_addoption`, `pytest_collection_modifyitems`, `pytest_runtest_makereport`; test-module `pytestmark` | Retained: pytest discovers these names dynamically. |
| `api.moments.GMMSandwich.finite_sample_correction_reason`, `estimation.covariance.shrinkage.ShrinkageResult.var_s_sum`, `bias_sq_sum` | Retained: fields on public result and configuration carriers. |
| `sample.packed.masked_row_mask` | Retained: documented public low-level operation in `docs/reference/method-catalogue.md`, independently usable to form ragged validity masks. Removal of its obsolete internal consumer does not retire this supported interface. |
| `estimation.survival.refinement.RefinementAttempt.train_log_likelihood`, `validation_log_likelihood`, `max_log_probability_gap`, `max_score_gap`, `parameter_shift_in_se` | Retained: named diagnostics in the public refinement attempt carrier. |
| `scripts.check_references.ReferenceResults.tests` | Retained: TypedDict inventory field returned by the reference-check script. |
| `api.embedding.PCoASpectrum.n_positive` | Retained: documented result metadata, populated by embedding construction. |
| `api.inputs._ArrowStream.__arrow_c_stream__.requested_schema` | Retained: parameter of the standard Arrow stream protocol signature. |
| `api.reporting` local Decimal context `prec` | Retained: setting precision is consumed implicitly by Decimal multiplication within `localcontext`; confidence-label regression tests cover high precision without changing the global context. |
| `api.plots._KindPredictor` | Retained: referenced by a quoted cast in `prediction_line`. |
| `core.axioms` properties `nonnegative`, `zero_diagonal`, `separates_points`, `triangle_inequality`, `strong_negative_type`; `core.matrices.squared_distance_negative_type` | Retained: public law/evidence properties. The separate deprecated composite alias was removed and independently reviewed. |
| `discrepancy.mmd.energy.geometry` and `geometry.ground.metrics` declarations `nonnegative`, `zero_diagonal`, `separates_points`, `triangle_inequality`, `strong_negative_type` | Retained: class-level law declarations, consumed through structural capabilities. |
| `geometry.ground._similarity_strategy` properties and declarations `positive_semidefinite`, `characteristic` | Retained: declared strategy law capabilities. |
| `geometry.barycentre.wasserstein.measure` result fields `initial_objective`, `constraint_violation`, `objective_semantics` | Retained: numerical diagnostics and semantics of the returned result. |
| `geometry.neighbour.NeighbourRanking` | Removed: absent from exports and source callers; corrected the stale ledger mention. Neighbour tests passed: 15. |
| `geometry.neighbour.NeighbourStabilitySummary.mean_top_k_overlap`, `exact_nearest_agreement`, `exact_nearest_agreement_share`; `NeighbourStabilityEdge.baseline_distance`, `comparator_distance`, `in_baseline_top_k`, `in_comparator_top_k` | Retained: documented public TypedDict output schema for external callers. |
| `inference.dyadic.contracts.point_design_names` | Retained: public report contract field populated in `inference.dyadic.report`. |
| `optimize.integration.gaussian.GaussianIntegrationResult.replicate_log_probabilities` | Retained: documented per-replicate diagnostic in the registered result pytree. |
| `optimize.psd` result fields `cond_before`, `cond_after`; `optimize.simplex` result field `complementarity_residual` | Retained: documented numerical diagnostics and dynamic pytree evidence. |
| `testing.qualification.bounds_method` | Retained: public qualification result metadata field. |
| `tests.api.test_dyadic._float64`, `tests.shapes.test_contracts._shape_environment`, `tests.testing.test_precision_references.record_precision` | Retained: pytest fixtures selected by autouse or fixture-name injection. |
| `tests.api.test_reporting.handle_data` | Retained: HTML parser callback invoked by `HTMLParser`. |
| `tests.core.test_axioms.__statadic_requires__` | Retained: dynamic contract metadata read by the axioms machinery. |
| Test metric declarations `nonnegative`, `zero_diagonal`, `separates_points`, `triangle_inequality` in `test_balancing`, `test_energy`, `test_metrics` | Retained: structural law declarations on test strategies. |
| `tests.geometry.test_neighbour.ticker` | Retained: deliberately invalid TypedDict field checked by the schema-leak regression. |
| `tests.migration.test_ground_discrepancy_migration.__call__` | Retained: callable-protocol implementation invoked by call syntax. |

The `pytestmark` entries throughout the test tree are pytest-discovered module
markers. Focused MMD tests passed: 177 passed,
3 skipped. An initial collection failure caused by orphaned decorators was fixed
before that successful rerun.

The deprecated balancing wrapper, solver aliases, linear effective-size mode,
chord-cost alias and composite axiom alias were removed, with canonical callers
retained. A different worker independently reviewed that the remaining algorithms
and normalization are unchanged. Their focused numerical suite passed 268 tests
with one skip, and three removed-solver rejection checks passed.

The covariance `var_S_sum` compatibility attribute was removed in favor of
`var_s_sum`; covariance tests passed 104 tests with six skips. Public diagnostic
fields remain supported evidence, rather than dead implementation paths.

A separate source-only Vulture scan exposed three private functions hidden in
the combined scan by test references or unrelated same-name fixtures. The
unused MMD `_clouds` and test-only `_mmd_squared_matrix_packed` were removed;
the traceability witness now exercises `mmd_squared_matrix`. The independent
`_rank_profile_greedy` reference moved to the design tests, preserving its
three parity uses. The affected MMD and design tests passed: 78 passed; scoped
Ruff lint and format checks passed.
