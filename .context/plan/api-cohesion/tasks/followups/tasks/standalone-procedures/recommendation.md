---
type: note
title: Recommended hosting decisions for standalone procedures
description: One recommended decision per inferential procedure outside the five workflow APIs, for the owner to accept or change.
links: []
---

# Recommended hosting decisions

Read at `00e7d8f` from source. This is a recommendation, not a decision: the owner
decides in the [task](README.md). Nothing here moves code or adds an adapter.
All paths are under `statadic.inference`. Parameter counts include keyword-only
parameters. Adapters retain the native result as `TestOutcome.evidence` and add no
calibration; each keeps its own law (U04, U05, U09). Mantel, PERMDISP and the
Sharpe joint stepdown are already fixed by those rows and are not listed as
adapter candidates.

| Procedure (module.function) | Returns | Params | Law it uses | Recommendation | Reason |
| --- | --- | --- | --- | --- | --- |
| `equivalence.equivalence_test` | `MixtureTestResult` | 17 | Permutation plus percentile bootstrap for `E(X, Y_w) >= delta`; docstring states no size result for selected weights | Stay standalone | Two p-values in opposite directions and selection settings in one call; one `TestOutcome` would hide one of them |
| `mixture.corrected_mixture_test` | `MixtureTestResult` | 15 | Studentized permutation of `X` against the selected mixture, weights re-optimized inside the null | Stay standalone | Mixes quantity, estimator and calibration; no single null `Proposition` states it |
| `resampling.permutation.energy_permutation_test` | `HypothesisTest` (typed form: `EnergyPermutationResult`) | 6 | Label permutation, add-one Monte-Carlo p-value over a supplied `EnergyGeometry` | Hypothesis adapter returning `TestOutcome` | Already carries a `CalibratedTest` and resampled null; `ResampledReference` (random, add-one) fits without new calibration |
| `sharpe.sharpe_difference_test` | `SharpeDifferenceTest` | 5 | Centered circular-block bootstrap of paired Sharpe difference | Hypothesis adapter returning `TestOutcome` | Two floats (statistic, p-value) against `H0: SR_a == SR_b`; only a proposition for equal Sharpes is missing |
| `sharpe.sharpe_ratio_ci` | `ConfidenceInterval` | 4 | Stationary-bootstrap percentile interval for one Sharpe ratio | Stay standalone (Estimand quantity only with a real consumer, U15) | Point value lives in `estimation.sharpe.sharpe_ratio`; the interval is a bootstrap summary, not a moment-based estimand |
| `sharpe.multiple_sharpe_test` | `MultipleSharpeResult` | 6 | Fixed-studentizer stepdown plus joint paired bootstrap | Stay standalone | Joint stepdown law is a recorded decision (U09); per-strategy vectors are not one test outcome |
| `sharpe.deflated_sharpe_ratio` | `DeflatedSharpeResult` | 4 | Asymptotic score against a supplied expected-maximum benchmark; no familywise control | Stay standalone | Docstring disclaims effective-count inference and finite-sample control; an adapter would imply a calibrated test |
| `gms.indicator_gms` | `GmsResult` | 8 | Uncorrected indicator GMS critical value by simulation, labelled `indicator_gms_uncorrected` | Stay standalone until a moment-inequality proposition exists | `api/hypotheses.py` has no inequality `Proposition`; reference must carry the uncorrected label |
| `wolak.wolak_test` | `WolakResult` | 4 | One-sided distance statistic with Monte-Carlo chi-bar-squared weights | Stay standalone until a moment-inequality proposition and chi-bar reference exist | `Reference` admits only `ChiSquare`, `FisherF`, `Normal`, `ResampledReference`; the weight-mixture tail fits none |
| `kappa.kappa_delta_method` | `KappaDeltaResult` | 3 | Delta-method variance propagation; no test | Stay standalone | Returns covariance-entry variances, not a statistic or p-value; Estimand covers supplied moments only (U01) |
| `resampling.bootstrap.bootstrap_confidence_interval` | `ConfidenceInterval` | 5 | Eager percentile bootstrap of a caller statistic | Stay standalone | Its docstring calls it a compatibility adapter; `bootstrap_estimate` is the transformable path |

## Order if the owner hosts any

1. `energy_permutation_test`, then `sharpe_difference_test`: the only rows needing no new proposition or reference type beyond one equality null.
2. Moment inequalities (`wolak_test`, `indicator_gms`) only together with a named inequality proposition and an honest reference; otherwise the shared `H0: E[g_t] >= 0` null would be forced into `LinearRestriction`.

## Not verified

- Whether `ResampledReference` (random, add-one) matches the Sharpe bootstrap p-value convention; read `sharpe_difference_test` p-value code before adapting.
- Median, barycentre and long-run-variance quantities named in U15: no standalone inferential function was reviewed for them here.
- Parity of any adapter with its native function (acceptance item three); none exists.
