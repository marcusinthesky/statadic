---
type: task
title: Type contrast estimands and rename the core estimand root
description: Rename core.statistics.Estimand to PopulationEstimand and give design contrasts a typed LinearPredictorContrast estimand.
links:
  - .context/plan/api-cohesion/tasks/estimand/README.md
status: done
---

# Type contrast estimands and rename the core estimand root

## Goal

`core.statistics.Estimand` is the zero-field root of the type-level estimand axis.
`api.estimand.Estimand` is the workflow host for covariance and supplied moments.
The shared name invites confusion, so the owner chose on 2026-11-11 to rename the
core root to `PopulationEstimand` and keep the workflow name (row F09).

Design contrasts also state what they target only as strings, in
`DesignContrast.estimand` and `ContrastResult.estimand`. The owner chose to type
them now: a `LinearPredictorContrast` record that belongs to the population axis.

## Acceptance

- [x] `core.statistics.PopulationEstimand` replaces the root `Estimand`. Subclass names and `EstimandT` are unchanged, and nothing in `src/` or the tests names the old root.
- [x] `LinearPredictorContrast(PopulationEstimand)` records `at`, `versus` and `scope`, and `describe()` reproduces today's strings exactly.
- [x] `DesignContrast.estimand` and `ContrastResult.estimand` take it. `RowDifference.compile` and `FittedDesign.contrast` build it, and contrast reports title themselves with `describe()`.
- [x] Tests cover both scopes, mapping immutability and report titles; existing contrast tests pass.
- [x] The named test files pass; lint passes for the touched files.

## Specification

- Files: `src/statadic/core/statistics.py`; the docstrings that name the root in
  `src/statadic/core/matrices.py`; `src/statadic/api/design.py`; `ContrastResult` and
  `contrast` in `src/statadic/api/inference.py`; the tests that assert on these (core
  statistics tests, `tests/api/test_design.py`, `tests/api/test_inference.py`,
  `tests/api/test_hypotheses.py`, `tests/api/test_regression_ux_hardening.py`); and
  `src/python/apps/docs/tests/test_external_extensions.py` if it compares estimand
  strings.
- Core: rename the class, its docstring ("Root of the population-estimand axis"),
  the bases of `DiscrepancyEstimand`, `DispersionEstimand` and
  `TransformedEstimand`, the `EstimandT` bound, `__all__` and comments. Use
  `git grep -n "\bEstimand\b" -- src/python/packages/statadic/src` and distinguish the
  core root from `api.estimand.Estimand`, which is unchanged.
- `api/design.py`:

  ```python
  @dataclass(frozen=True)
  class LinearPredictorContrast(PopulationEstimand):
      """The linear-predictor difference between two predictor settings."""

      at: Mapping[str, float | int | str]
      versus: Mapping[str, float | int | str]
      scope: Literal["rows", "term"]

      def describe(self) -> str: ...
  ```

  - Freeze the mappings with `MappingProxyType(dict(...))`, as `RowDifference`
    does.
  - `scope="rows"` describes as `f"linear predictor at {dict(self.at)} versus {dict(self.versus)}"`.
  - `scope="term"` holds single-entry mappings `{column: value}` and describes as
    `f"structural linear-predictor contrast: {column}={at} versus {versus}"`.
  - Both strings must match today's exactly.
  - `DesignContrast.estimand: LinearPredictorContrast`.
- `api/inference.py`: `ContrastResult.estimand: LinearPredictorContrast`, and
  `report()` uses `self.estimand.describe()` as its title. Leave every other part
  of `inference.py` alone.
- Do not touch `src/lean/registry`, the catalogue or the docs pages; the
  coordinator and `fit-cleanup-docs` own them.

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
