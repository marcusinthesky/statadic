# Survival models: choose a supported problem

This guide introduces the supported survival problems. Consult the full [survival workflow](../survival-models.md) for details.

## Ordinary accelerated failure time models

The host family supports Weibull, lognormal and log-logistic AFT models for right-censored **elapsed durations**. A row is one spell: `event=1` records an event; `event=0` records right censoring. Durations must be positive and finite. Optional delayed entry obeys `0 ≤ entry < duration` and conditions on survival to entry.

This is not a start–stop counting-process interface. `SurvivalML` requires an explicit family. Weibull and log-logistic AFT use log shape as their ancillary parameter; lognormal AFT uses log standard deviation. These parameters have different interpretations even when they occupy the same position in a fitted vector.

`predict(data)` returns marginal medians. Select `kind="survival"`, `"cumulative_hazard"`, `"quantile"`, `"mean"` or `"restricted_mean"` explicitly for another target. Mean prediction returns one unconditional duration mean per observation and accepts no time, grid or entry-conditioning argument. Log-logistic shape at or below one has no finite mean and returns NaN without invalidating other predictions. Log-logistic RMST uses bounded quadrature; unresolved numerical estimates are NaN, and successive agreement is not an error certificate.

A scalar broadcasts over observations; a vector is row-aligned. Pass `grid=True` for an observation-by-value grid from a one-dimensional time/horizon array. Model-based inference uses observed information and exposes availability. It does not itself establish finite-sample calibration.

## Repeated spells

The panel layout permits unbalanced entity-period observations with at most one spell per entity-period and selected pooled/unit/time/two-way effects. Unit, time and two-way fixed effects reject an explicit `Intercept()` term. Coding is retained for prediction, and new fixed-effect levels are rejected.

Fixed effects do not add frailty or model repeated-spell dependence automatically. Disconnected incidence can still produce a rank-deficient design under the retained normalization. Ordinary iid AFT qualification does not certify this repeated-spell inference.

## Spatial log-time survival

This is a distinct, more demanding model with lognormal errors, graph-dependent log times and numerical integration. Keep covariates for all active graph nodes separate from observed spell outcomes. Missing outcomes are integrated out, not encoded as zero durations. Positive spatial entry is currently rejected by the host.

Supply one `SpatialSurvivalData(panel_data, integrations)` input to the model.
Preparation retains its graph alignment and fixed integration designs. The expert
`refine(prepared, validation_integrations=..., config=...)` utility compares
independent designs without moving replayable draws onto the estimator.

Predictions are unconditional marginal node distributions in graph order, not distributions conditioned on observed neighbours' durations. The mean uses each node's propagated log-time variance, rather than just its innovation variance. Spatial survival remains lognormal-only. Host preparation and refinement are eager. Independent training/validation integration streams and retained refinement attempts provide stability evidence—not an error bound, a global optimum certificate or statistical qualification.

> **Read before fitting:** the validation guide records adverse high-censoring spatial-integration fixtures and two inconclusive ordinary-AFT qualification cells. See [known limitations](../validation/known-limitations.md). Do not label the survival family “fully validated”.

## What a completed tutorial must report

Show the censoring/entry coding, family and ancillary convention, event counts, fit/inference availability and the meaning of each prediction scale. For spatial survival, also report node/outcome counts, graph ordering, integration design, refinement disposition and applicable numerical stress results.
