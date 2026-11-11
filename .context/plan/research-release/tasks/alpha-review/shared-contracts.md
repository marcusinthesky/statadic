# Shared-contract adjudication

Both passes and independent challenge completed for the first thirteen source
diffs listed in [coverage](coverage.md). This is a bounded result, not closure of
the entire shared-contract slice. No production source changed.

## Keep

Retain the separation of models, estimator bricks, native fitted evidence,
family-specific inference and eager host preparation (D02, D06, D07). Current
consumers justify these boundaries. `ParameterResult` serves non-regression
families; merging it with `RegressionResult` would impose learned-design state
on unrelated results.

Keep `DMat`, `SpaceDMat`, `PairwiseEstimateMatrix`, equation and variability
descriptors, scalar provenance, transformed spaces and authority markers where
they express different contracts. Signed estimates cannot become dissimilarity
matrices merely because both have two axes. Runtime workflows cannot recover
mathematical evidence erased from provenance (F004).

Keep numerical QR influence calculations separate from generic sandwich
composition. Their solver/rank choices have scientific and numerical meaning.
Keep `typed_vmap` and `map_arrays`; resampling and transport use them. Host
labels, role validation, retained designs and report construction are eager by
contract and do not need artificial tracing wrappers.

## Proposed alpha cuts and consolidation

| ID | Disposition | Reason and acceptance |
| --- | --- | --- |
| S001 | Merge duplicated covariance assembly. | Route generic limiting covariance through existing `_parameter_covariance_from_limiting`, preserving its input checks, panel counts and rank semantics. The helper already serves linear/GMM/panel/IV/spatial producers and checks output PSD; this also addresses F003. |
| S002 | Remove `core.typing.DMat` and `core.LegacyDMat`. | No production consumer beyond the reexport was found; only legacy tests/docstrings remain. Explicit alpha import break. Transfer useful rejection tests to modern carriers and regenerate inventories explicitly. Retain the active nominal law chain. |
| S003 | Remove `LazyFrame` from accepted public input types. | Only its definition and union arm use the protocol; runtime always rejects uncollected lazy frames. Keep the helpful runtime rejection. |
| S004 | Defer four unused atomized pullbacks. | `pullback_nonnegative`, `pullback_zero_diagonal`, `pullback_symmetric`, `pullback_triangle` have definitions/exports but no shipped consumer, extension test or claim. Keep used `pullback_pseudometric`, `pullback_negative_type`, restriction and quotient operations. |

These are coordinator recommendations for the repair backlog. They are not
implemented or accepted as safe solely from a dead-code scanner. Recheck
artifacts, registries, downstream examples and public break documentation at the
corrective-source freeze.

## Execution and performance analysis

The bounded smooth-interior probes passed value, reverse gradient, JVP,
forward-over-reverse HVP and Hessian controls for chunked mapping and covariance
composition. Changed array values reused stable callables. No solver-solution
derivative, rank-boundary derivative or universal host transformation claim
follows. No timing or memory superiority was measured.

Profile these retained workload candidates after corrective-source freeze:

- Mapping preparation copies every unused NumPy column for snapshot safety;
  compare explicit selected observations and record retained memory separately.
- Spline preparation synchronizes minimum and maximum separately; measure
  preparation apart from repeated transformation.
- Cyclic basis construction creates an extra reference column before slicing;
  inspect lowered allocations before asserting a material cost.
- Generic covariance diagnostics perform rank, PSD and pseudoinverse work;
  inspect optimized compiled operations before claiming duplicate execution.

F001 and F002 violate intended interval availability and frozen-input contracts.
F003 and F004 are inherited alpha risks. Reviewers independently challenged all
four; the coordinator reproduced their bounded witnesses. Source/proof review
found no invalid theorem in the two inspected Lean claims. Those theorems assume
the corresponding generating feature map and injectivity; the Python witness
composition does not establish those assumptions automatically.
