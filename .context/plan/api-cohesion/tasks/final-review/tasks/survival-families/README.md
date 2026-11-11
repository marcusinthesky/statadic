---
type: task
title: Move survival families to likelihood summaries
description: Make AFT and spatial survival results supply a LikelihoodSummary and import fit statistics from their owner.
links:
  - .context/plan/api-cohesion/tasks/final-review/tasks/fit-contract/README.md
status: done
---

# Move survival families to likelihood summaries

## Goal

This applies the [fit-contract](../fit-contract/README.md) pattern to the survival
results.

## Acceptance

- [x] The AFT and spatial survival results implement `likelihood_summary()` (`k` = parameter width, `n` = spells), and their `fit_statistics` overrides are deleted.
- [x] `api/survival/inference.py` imports `FitStatistics` from `statadic.api.fit_statistics`.
- [x] Metric values, names, order and notes are unchanged; the tests assert the summaries and static `SummarizesLikelihood` conformance.
- [x] `tests/api/survival/test__model.py`, `test__spatial.py` and `test_inference.py` pass, along with the survival external references; lint passes.

## Specification

- Files: `src/statadic/api/survival/{_model,_spatial,inference}.py` and
  `tests/api/survival/{test__model,test__spatial,test_inference}.py`.
- Move each counting docstring onto `likelihood_summary`, and keep the family
  reason strings.

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
