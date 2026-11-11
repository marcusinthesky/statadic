---
type: task
title: Own fit statistics in one typed module
description: Move fit statistics out of the lifecycle module into api.fit_statistics with a typed likelihood summary, canonical metric names and structural capabilities.
links:
  - .context/plan/api-cohesion/tasks/final-review/tasks/fit-contract/specification.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: done
---

# Own fit statistics in one typed module

## Goal

`api/base.py` owns the `Model` and `Result` lifecycle. It also holds
`FitStatistics`, `information_metrics` and least-squares mathematics, and families
import its private `_INFORMATION_NOTE` and `_linear_fit_statistics`. Each family
hand-counts `k` and `n` inside its own `fit_statistics`. This task gives fit
statistics one owner, `statadic.api.fit_statistics`, with a typed `LikelihoodSummary`
that families supply (row F08). It also accepts the review's recommendation to
move family mathematics out of the lifecycle owner. The helper cannot live in
`api.regression`, because `api.regression` depends on `api.spatial`, and the
spatial SLX route also needs it.

Families migrate in wave 3. This task therefore keeps transitional re-exports in
`api/base.py`, which `fit-cleanup-docs` removes.

## Acceptance

- [x] `statadic.api.fit_statistics` holds `LikelihoodSummary`, `FitStatistics` with `from_likelihood`, `information_metrics`, `gaussian_least_squares_summary` and `linear_fit_statistics`, with unchanged numerical results.
- [x] `Result.likelihood_summary()` defaults to `None`, and the default `Result.fit_statistics()` derives from it.
- [x] `FitStatisticName` and `FIT_STATISTIC_NAMES` in `api.reporting` name the standard metrics in specification order, and the TeX label map is keyed by them.
- [x] The `ReportsFitStatistics` and `SummarizesLikelihood` Protocols are in `api.capabilities`, with static conformance for every family result that reports fit statistics.
- [x] Tach declares the module and its consumers, and `just python::architecture` passes.
- [x] The named test files pass; lint passes for the touched files.

## Specification

The [specification](specification.md) is normative.

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

- [Remove transitional fit imports and document the contracts](../fit-cleanup-docs/README.md)
- [Move regression families to likelihood summaries](../regression-families/README.md)
- [Move spatial families to likelihood summaries](../spatial-families/README.md)
- [Move survival families to likelihood summaries](../survival-families/README.md)

<!-- insitu:end -->
