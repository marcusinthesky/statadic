---
type: task
title: Move spatial families to likelihood summaries
description: Make spatial cross-section, graph, observed-panel and panel results supply a LikelihoodSummary and delete the private spatial statistics helper.
links:
  - .context/plan/api-cohesion/tasks/final-review/tasks/fit-contract/README.md
status: done
---

# Move spatial families to likelihood summaries

## Goal

This applies the [fit-contract](../fit-contract/README.md) pattern to the spatial
results.

## Acceptance

- [x] `SpatialRegressionResult`, `GraphRegressionResult`, the observed-panel result and `SpatialPanelResult` implement `likelihood_summary()`, with `k` and `n` as their current docstrings count them.
- [x] `_likelihood_statistics` is deleted; SLX least squares keeps `linear_fit_statistics`; moment, GMM and KKP results return `None`.
- [x] No module imports private fit-statistics names from `api.base`.
- [x] Metric values, names, order and notes are unchanged; the tests assert the summaries and static `SummarizesLikelihood` conformance.
- [x] `tests/api/test_spatial.py`, `tests/api/test_spatial_graph_regression.py` and the spatial external walkthroughs pass; lint passes.

## Specification

- Files: `src/statadic/api/spatial/{cross_section,graph_regression,observed,panel}.py`,
  `tests/api/test_spatial.py` and `tests/api/test_spatial_graph_regression.py`.
- The log likelihood is `float(core.log_likelihood)` when the fit is valid, else
  NaN. The reason is `Spatial Gaussian log likelihood needs a valid fit.`
- Move each `fit_statistics` counting docstring onto `likelihood_summary`.
- `SpatialRegressionResult`:
  - A `LinearFit` core keeps a `fit_statistics` override returning
    `linear_fit_statistics(core, weighted=False)`, and its summary is
    `gaussian_least_squares_summary(core)`.
  - A non-likelihood core returns a `None` summary.
- Import from `statadic.api.fit_statistics`.

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
