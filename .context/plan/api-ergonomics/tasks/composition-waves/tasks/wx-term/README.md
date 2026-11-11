---
type: task
title: Share WX lag selection, then decide the WX design term
description: "Share named WX lag selection across spatial hosts (done), then decide whether exogenous WX becomes a prepared, graph-aware design term usable by ordinary GLMs."
links: []
status: backlog
---

# Share WX lag selection, then decide the WX design term

## Goal

Gaussian spatial hosts and spatial survival each resolved `lag_columns`
separately. An exogenous neighbour covariate WX is also the most useful
spatial feature for Poisson and logistic models, but today WX is an estimator
capability (positional `slx_columns` on each process), not a design term.

## Done in wave 1

- `statadic.api.panel.select_lag_columns` is the one named-WX selector. It is used
  by `_named_lags` (cross-section, panel and observed-panel hosts) and by
  `SpatialSurvivalRegression`, which still offers only design columns, not
  fixed-effect columns, by name.

## Open decisions

1. **Design term or estimator capability.** A `SpatialLag` term would follow
   `DistanceToLandmarks` (`api/design.py`): retain the graph and unit column at
   `prepare`, align by IDs in `transform`. `FittedTerm.transform` receives only
   `Observations`, so the term keeps the graph in its fitted state, or the
   signature gains a context argument.
2. **Prediction roster.** WX mixes rows. Prediction must cover the full graph
   roster or accept a replacement graph with its own IDs; subset prediction,
   which every other term allows, breaks.
3. **Spatial ML pairing.** Impacts, covariance and the LR test need to know
   which columns lag which. Moving the spatial bricks from `slx_columns` to a
   term touches about six estimator modules plus inference; a GLM-only term
   leaves them alone.
4. **Persistence.** `api/artifacts/_design.py` has a closed term registry; a new
   fitted term needs a codec (graph arrays plus unit IDs) or an explicit
   `save_predictor` rejection.
5. **Operator order.** With changing graphs, lagging before or after effect
   removal changes identifying variation; the term must name its order.

## Acceptance

- [x] One shared named-WX selector for every spatial host.
- [ ] Decide decisions 1–5 and record them here.
- [ ] If a term is chosen: Poisson and logistic fits with WX, ID-aligned
  prediction, artifact codec or rejection, and the zero-graph reduction test.

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
