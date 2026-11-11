---
type: task
title: Declare exports in the measure and geometry families
description: "Explicit __all__ for core, sample, optimize, geometry, discrepancy, association and operators, keeping every name pricing-perspective imports public."
links: [".context/plan/public-surface/handoff.md", ".context/plan/public-surface/evidence.md"]
status: done
---

# Declare exports in the measure and geometry families

## Goal

Make the shared contract, sample, optimizer, geometry, discrepancy, association and operator families follow the plan's rule. pricing-perspective relies on these modules directly, so treat every name it imports as public.

## Acceptance

- [x] Every module listed below declares a literal `__all__`.
- [x] Each unlisted name below is listed or underscored, decided by the [handoff](../../handoff.md) procedure.
- [x] Every name that pricing-perspective imports from these families remains public.
- [x] `verification.md` records each decision and the coordinator requests.
- [x] The handoff checks for export tasks pass.

## Owns

Under `src/python/packages/statadic/src/statadic/`:

- `core/**`
- `sample/**`
- `optimize/**`
- `geometry/**`
- `discrepancy/**`
- `association/**`
- `operators/**`

It also owns the mirrored directories under `tests/`.

## Work list

**Enforcement follow-up:** combine the existing literal plus augmented export lists
in `statadic.core` into one literal declaration, preserving the exact export set.

**Navigation namespaces**, which take `__all__ = []` unless they already re-export something deliberately:

- `statadic.association`
- `statadic.discrepancy`
- `statadic.geometry`
- `statadic.geometry.barycentre`
- `statadic.operators`
- `statadic.sample`

**Leaf modules without `__all__`:**

| Module | Public definitions |
| --- | --- |
| `statadic.discrepancy.mmd.energy.balancing.effective_size` | 1 |
| `statadic.discrepancy.mmd.energy.balancing.mixture` | 1 |
| `statadic.discrepancy.mmd.energy.balancing.statistics` | 2 |
| `statadic.discrepancy.mmd.energy.balancing.weights` | 2 |
| `statadic.discrepancy.mmd.energy.weighted` | 2 |
| `statadic.discrepancy.mmd.weighted` | 4 |
| `statadic.discrepancy.transport.matrix` | 5 |
| `statadic.geometry.barycentre.mmd.solve` | 2 |
| `statadic.optimize.integration.gaussian` | 6 |
| `statadic.sample.alternatives` | 2 |
| `statadic.sample.packed` | 7, imported by pricing-perspective |
| `statadic.sample.reduction` | 2, imported by pricing-perspective |

**Unlisted definitions in modules that already declare `__all__`:**

| Module | Names |
| --- | --- |
| `statadic.association.decomposition` | `DiscoIndexProvenance`, `DiscoFRatioProvenance` |
| `statadic.core.typing` | `static_jaxtyping` (runtime branch; omitted from original measurement) |
| `statadic.core.domains` | `NormalizationPullbackOrigin`, `UnitNormSupport` |

Provenance and domain types often appear in public result annotations; check rule 5 first.

The core private helpers that are already exposed through scoped Tach interfaces stay as they are:

- `matrices._dmat`, `matrices._pairwise_estimate` and `matrices._space_dmat`;
- `equations._variability_matrix`, `equations._parameter_covariance_from_limiting` and `equations._symmetric_psd`;
- `backends._least_squares_factored`.

## Public signature contracts found during review

Declare `UStatisticEstimates`, `VStatisticEstimates` and `RawIdentityTransform`
public at their defining modules. Re-export the existing `SimilarityFunction`
from the public modules exposing `declare_similarity`. Preserve underlying
alias definitions. Constant-only annotation aliases stay private.

## Non-goals

- No change to `PackedClouds` or any other carrier.
- No module moves.
- No change to numerical code.

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
