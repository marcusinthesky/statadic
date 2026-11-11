---
type: task
title: Declare the host API exports
description: "Give every statadic.api module an explicit __all__, and list or underscore each public-looking definition."
links: [".context/plan/public-surface/handoff.md", ".context/plan/public-surface/evidence.md"]
status: done
---

# Declare the host API exports

## Goal

Make the `statadic.api` host workflow layer follow the plan's rule. Every public module declares `__all__`, and every public-looking definition is either listed or underscored.

## Acceptance

- [x] `statadic.api`, `statadic.api.reporting` and `statadic.api.representation` declare a literal `__all__`.
- [x] Each name listed below is either in `__all__` or underscored, decided by the [handoff](../../handoff.md) procedure.
- [x] `verification.md` records each decision and the coordinator requests.
- [x] The handoff checks for export tasks pass.

## Owns

- `src/python/packages/statadic/src/statadic/api/**`
- `src/python/packages/statadic/tests/api/**`
- `src/python/packages/statadic/tests/integration/api/**`

## Work list

**Modules without `__all__`:**

| Module | Note |
| --- | --- |
| `statadic.api` | Navigation namespace: `__all__ = []` |
| `statadic.api.reporting` | 11 public definitions |
| `statadic.api.representation` | 10 public definitions |

**Settled for `statadic.api.reporting`:**

- `freeze_metadata` and `metadata_to_json` are public.
- `_confidence_label` keeps its Tach exception.

**To decide in `statadic.api.reporting`:**

- `finite_metric` is used only inside `statadic.api`.
- `render_pgfplots` is used only by tests.

**Unlisted definitions in modules that already declare `__all__`:**

| Module | Names |
| --- | --- |
| `statadic.api.embedding` | `RowIds`, `PreparedPCA`, `PreparedPCoA` |
| `statadic.api.association` | `MantelInput`, `PreparedMantel` |
| `statadic.api.covariance` | `CovarianceContext`, `PreparedCovariance` |
| `statadic.api.data` | `NumericInput`, `check_array_precision` |
| `statadic.api.inputs` | `InputRole`, `model_roles` |
| `statadic.api.dyadic` | `PreparedDyadic` |
| `statadic.api.tables` | `ArrowFrame` |

`Prepared*` types and input aliases are candidates for rule 5. Check whether they appear in public `prepare`, `fit` or `predict` signatures before underscoring them.

## Public signature contracts found during review

Also expose `DtypedColumn`, `InferenceContext`, `Predictor`, `KindPredictor` and
`RegressionFit` from their public owning modules. Re-export `SpatialEstimator`,
`SpatialCoreFit`, `InstrumentRoles`, `SLXColumns`, `GroupLabels`, `RowIds`,
`PreparedIV`, `PreparedObservedSpatialPanel`, `ComparisonCore`,
`NativeDecomposition` and `ComparisonTest` from their public workflow families.
Expose the latter's four comparison/calibration aliases and five origin classes
so its retained provenance types are public too.
These are existing signature, generic-bound, result or user-implemented contracts.

## Non-goals

- No new host API.
- No changes to how any result, report or artifact behaves.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
