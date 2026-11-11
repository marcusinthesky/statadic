---
type: task
title: Label standard fit statistics in LaTeX tables
description: Give the package's canonical fit-statistic metric names manuscript labels in LaTeX tables, overridable by the caller.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: done
---

# Label standard fit statistics in LaTeX tables

## Goal

Metric names such as `n_observations` print as identifiers in LaTeX tables. The
package defines the names of its standard fit statistics, so `to_latex` can set
manuscript labels for them by default. The [specification](../table-content/specification.md)
lists the names.

## Acceptance

- [x] `to_latex` labels `n_observations` Observations, `n_parameters` Parameters, `r_squared` `$R^2$`, `adjusted_r_squared` Adjusted `$R^2$`, `uncentred_r_squared` Uncentred `$R^2$`, `log_likelihood` Log likelihood, and `aic`, `bic` and `hqic` in capitals.
- [x] A caller's `labels=` entry overrides a default; text, HTML and value keys keep the retained names.
- [x] Notes that name a labelled metric use its label.
- [x] The docs report compile test covers a footer with every default label; `tests/api/test_reporting.py` and the docs reporting tests pass; lint and strict typing pass.

## Specification

- File: `src/python/packages/statadic/src/statadic/api/reporting.py` only, with its tests
  and the docs reporting compile test. The TeX labels are package constants and
  are inserted unescaped; caller labels keep their `label_markup` rule.

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

- [Select fit statistics in LaTeX table footers](../../../final-review/tasks/latex-footer-selection/README.md)

<!-- insitu:end -->
