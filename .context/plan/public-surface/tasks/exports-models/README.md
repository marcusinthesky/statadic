---
type: task
title: Declare exports in the estimation, inference and model families
description: "Explicit __all__ for estimation, inference and model modules, the dyadic contracts pricing-perspective uses, and a public home for ProjectionUnavailableError."
links: [".context/plan/public-surface/handoff.md", ".context/plan/public-surface/evidence.md"]
status: done
---

# Declare exports in the estimation, inference and model families

## Goal

Make the estimation, inference and model families follow the plan's rule. This includes the dyadic contract modules that pricing-perspective uses most heavily, and the one public exception that currently lives only in a private module.

## Acceptance

- [x] Every module listed below declares a literal `__all__`, and the retired `inference/dyadic/fit.py` is deleted.
- [x] Each unlisted name below is listed or underscored, decided by the [handoff](../../handoff.md) procedure.
- [x] `ProjectionUnavailableError` is public from the module whose calls raise it. `apps/studies/src/statadic_studies/adapters/bounds.py:9` imports it from there, and the coordinator removes its Tach exception (`tach.toml:1255-1259`).
- [x] Every name that pricing-perspective imports from these families remains public.
- [x] `verification.md` records each decision and the coordinator requests.
- [x] The handoff checks for export tasks pass, plus `just python::studies::check`.

## Owns

Under `src/python/packages/statadic/src/statadic/`:

- `estimation/**`
- `inference/**`
- `model/**`

It also owns the mirrored directories under `tests/`, and the single import line in `apps/studies/src/statadic_studies/adapters/bounds.py`.

## Work list

**Navigation namespaces**, which take `__all__ = []` unless they already re-export something deliberately:

- `statadic.estimation`
- `statadic.inference`
- `statadic.inference.resampling`
- `statadic.model`

**Leaf modules without `__all__`:**

| Module | Public definitions |
| --- | --- |
| `statadic.estimation.dyadic.contracts` | 2 |
| `statadic.estimation.dyadic.fit` | 2 |
| `statadic.estimation.dyadic.solve` | 1 |
| `statadic.estimation.spatial.observed_panel` | 4 |
| `statadic.inference.clustering` | 1 |
| `statadic.inference.contracts` | 3 |
| `statadic.inference.dyadic.bootstrap` | 4 |
| `statadic.inference.dyadic.contracts` | 6 |
| `statadic.inference.gms` | 2 |
| `statadic.inference.sharpe.deflated` | 4 |
| `statadic.inference.sharpe.difference` | 4 |
| `statadic.inference.sharpe.multiple` | 4 |
| `statadic.inference.survival.information` | 3 |
| `statadic.inference.wolak` | 10 |
| `statadic.model.dyadic.specification` | 3 |
| `statadic.model.dyadic.validation` | 6 |
| `statadic.model.survival.spatial` | 1 |

**Unlisted definitions in modules that already declare `__all__`:**

| Module | Names |
| --- | --- |
| `statadic.inference.mantel` | `Float`, `Index`, `mantel_bootstrap_kernel` |
| `statadic.inference.resampling.permutation.test` | `StatisticFn`, `EnergyStatisticProvenance`, `EnergyLabelPermutation` |
| `statadic.inference.resampling.block_length` | `BlockStatistic`, `EagerBlockStatistic` |
| `statadic.inference.resampling.bootstrap` | `BootstrapStatistic`, `EagerBootstrapStatistic` |
| `statadic.estimation.covariance.shrinkage` | `ShrinkageWeight` |
| `statadic.estimation.likelihood` | `Dispersion` |
| `statadic.estimation.spatial.likelihood` | `SpatialModel` |
| `statadic.inference.panel_covariance` | `PanelEquation` |
| `statadic.inference.spatial.covariance` | `SpatialEquation` |

Statistic protocols that users pass to bootstrap or permutation calls fall under rule 5.

**Retired module:** `statadic/inference/dyadic/fit.py` contains only the docstring "Retired module: the combined dyadic fit door moved into composition by callers." Nothing imports it, in STATADIC or in pricing-perspective. Delete it as a clean break; `docs/module-organization.md:54` already records the replacement call.

## Public signature contracts found during review

Declare `FloatScalar` and `ReturnVector` in `inference.sharpe.difference`;
`FloatScalar`, `ReturnVector`, `StrategyVector` and `StrategyPanel` in
`inference.sharpe.multiple`; and `FixedGraphCache` in `estimation.spatial.panel`.
These are existing public signature/result-field aliases; preserve their exact
underlying types. Private-signature bootstrap aliases stay private.

## Non-goals

- No change to `statadic.inference._common` beyond what the re-export needs.
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
