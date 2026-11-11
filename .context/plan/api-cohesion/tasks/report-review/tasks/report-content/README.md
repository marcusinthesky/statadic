---
type: task
title: Repair report counts, verbs and keywords across producers
description: Make report producers retain counts as integers, show the sample size on inference tables, and use one report verb and one confidence keyword.
links:
  - .context/plan/api-cohesion/tasks/report-review/audit.md
status: done
---

# Repair report counts, verbs and keywords across producers

## Goal

Producers under `src/python/packages/statadic/src/statadic/api/` build each `Report`.
Rows P04 and P05 of the [dispositions](../../../../proposal-dispositions.md)
accept four repairs to what they retain and how they are called. The
[audit](../../audit.md) holds the measurements.

`api/reporting.py` belongs to `latex-tables` and is not edited here. No dataclass
field changes and no new statistics.

## Acceptance

- [x] Every `Metric` that is a count by construction retains an `int`: observations, clusters, components, ranks, axes, draws and integral degrees of freedom. Real-valued metrics stay `float`.
- [x] `InferenceResult.report` retains `Metric("n_observations", <int>)`, so every renderer shows the sample size. Its metadata is unchanged.
- [x] A test component with nothing to define has an empty definition, not `Test component.`
- [x] `GraphBootstrap.report` takes `confidence=`, not `level=`; its guide and tests follow.
- [x] `summary()` is removed from all nine classes that define it as `return self.report()`; callers, tests and the package `AGENTS.md` use `report()`.
- [x] An integer count survives `save_analysis` and `load_analysis` as an `int`.
- [x] The named test files for every touched module pass; lint and strict typing pass for the touched files.

## Specification

- Find counts by reading every `Metric(` construction under `api/`. Convert at the
  producer with `int(...)`; do not round a value that can be fractional.
- `summary()` is defined in `api/base.py`, `api/inference.py`, `api/hypotheses.py`,
  `api/diagnostics.py`, `api/dyadic.py` (twice), `api/comparison/_results.py`
  (twice) and `api/regression/_conditional.py`. `estimate` is the only estimation
  verb and `report` the only reporting verb; add no alias or deprecation shim.
- Existing tests that call `.summary()` are rewritten to `.report()`, or deleted
  where they only assert the alias.
- Do not edit the public inventory, statement pins or `catalogue.json`. The
  coordinator reconciles removed and added symbols once.
- Tests in `tests/api/test_reporting.py` belong to `latex-tables`. If one fails
  because of this task, report it instead of editing it.

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
