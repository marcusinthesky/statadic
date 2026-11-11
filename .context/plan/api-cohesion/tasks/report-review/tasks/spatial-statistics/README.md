---
type: task
title: Report information criteria for spatial likelihood fits
description: Report n_parameters, log likelihood, AIC, BIC and HQIC for spatial cross-section, observed-graph, graph, panel and spatial survival likelihood fits.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: done
---

# Report information criteria for spatial likelihood fits

## Goal

Extend the fit statistics of [table-content](../table-content/README.md) to spatial
likelihood fits, so their criteria compare with non-spatial fits of the same
response. Start when `table-content` is done. The
[specification](../table-content/specification.md) is normative.

## Acceptance

- [x] Spatial cross-section, observed, graph-regression, spatial panel and spatial survival likelihood results override `fit_statistics()` with the shared helper, and their `infer` passes it on. Non-likelihood estimators report none.
- [x] Each log likelihood includes the Jacobian and Gaussian constants; the test file states `k` per family, counting spatial parameters and the scale.
- [x] External-reference tests match spreg ML lag and ML error `logll`, and `aic` and `schwarz` through spreg's stated offset; other spatial families state why no peer exists.
- [x] The named test files for the touched modules pass; lint and strict typing pass for the touched files.

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

- [Compare spatial panel and SAC likelihoods with references](../../../final-review/tasks/spatial-likelihood-references/README.md)

<!-- insitu:end -->
