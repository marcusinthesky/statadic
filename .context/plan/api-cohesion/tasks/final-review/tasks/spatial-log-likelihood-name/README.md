---
type: task
title: Name every native log likelihood log_likelihood
description: Rename the spatial native fit field loglik to log_likelihood everywhere and describe likelihood fits with one Protocol.
links: []
status: done
---

# Name every native log likelihood log_likelihood

## Goal

GLM, generalised-likelihood and survival fits name their maximised log likelihood
`log_likelihood`. The spatial cross-section, graph, observed-panel and panel fits
name it `loglik`. One name, and one Protocol that names it, lets fit summaries
treat every likelihood fit alike (row F06). The change is a rename only: nothing
numerical changes.

## Acceptance

- [x] No native fit field or reader named `loglik` remains under `src/statadic`, the tests or `src/python/apps` (excluding `apps/performance/history/**`).
- [x] `LikelihoodFit(Protocol)` in `statadic.inference.contracts` names `log_likelihood` and `valid`, and a static conformance test covers the GLM, generalised-likelihood, AFT and spatial fits.
- [x] Field order and numerical results are unchanged.
- [x] The named test files for the touched modules pass, including the spreg references on the external lane; lint passes for the touched files.

## Specification

- Files:
  - `src/statadic/estimation/spatial/*`, `src/statadic/estimation/survival/spatial.py`,
    `src/statadic/inference/spatial/diagnostics.py`, `src/statadic/inference/contracts.py`.
  - Only the `core.loglik` reader lines in
    `src/statadic/api/spatial/{cross_section,graph_regression,observed,panel}.py`.
  - Tests under `tests/estimation/spatial/`, plus
    `tests/estimation/test_spatial_reference.py`, `tests/api/test_spatial.py`,
    `tests/api/test_spatial_graph_regression.py`,
    `tests/api/spatial/test_learning.py`, `tests/api/regression/test__model.py`,
    `tests/integration/api/test_spatial_reference_walkthroughs.py`,
    `tests/qualification/test_sar_diagnostics.py`, and the mirrored test of
    `inference/contracts.py`.
  - Readers in `src/python/apps/performance/src/**`, `src/python/apps/studies/**`
    and `src/python/apps/docs/src/statadic_docs_labs/consumer_fixtures.py`.
- Leave `apps/performance/history/**` untouched; it holds archived measurement
  scripts.
- Find readers with `git grep -n "loglik\b"`. Distinguish the fit field and its
  readers, which are renamed, from local variables or peer attributes such as
  spreg's `logll`. Rename local variables only where they mirror the field.
- Protocol:

  ```python
  class LikelihoodFit(Protocol):
      @property
      def log_likelihood(self) -> Array: ...
      @property
      def valid(self) -> Array: ...
  ```

  Add it to `__all__`. The conformance test assigns real fitted instances, or
  annotated parameters, to `LikelihoodFit` so that pyrefly checks them.
- Do not touch `src/lean/registry`, the catalogue or the inventory; the
  coordinator owns them.

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
