---
type: task
title: Require only the columns each prediction kind uses
description: "Make each prediction kind need only the columns it uses, with one rule for live and reloaded predictors."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Require only the columns each prediction kind uses

## Goal

Each prediction kind must require only the columns it uses, and live and reloaded predictors must accept the same kinds under the same rule.

## Acceptance

- [x] Rate prediction with no exposure column works on live and reloaded Poisson predictors.
- [x] The role table below holds for Poisson, logistic, NB2 and probit on live and reloaded predictors where artifacts exist.
- [x] A supplied exposure is still validated as positive when `kind="rate"`.
- [x] A missing required role raises `KeyError` naming the column, role and kind.
- [x] Columns used by design terms or a declared offset stay required for every kind.
- [x] `kind="linear"` is removed from artifacts, and `LinearResult.predict` gains `kind="mean"`.
- [x] A parity test in `tests/api/test_regression_artifacts.py` covers every family and kind.
- [x] Docs and `CHANGELOG.md` are updated, and `just check` passes.

## Specification

| Family | Kind | Needed beyond design and offset |
| --- | --- | --- |
| Poisson, NB2 | `rate` | Nothing |
| Poisson, NB2 | `mean`, `link` | Exposure, when declared |
| Logistic, probit | `probability`, `link` | Nothing |
| Logistic, probit | `mean` | Trials, when declared |
| Linear, Gamma | All | Nothing |

Sites to change:

- `api/regression/_glm.py:75` and `:177`
- `api/regression/_likelihood.py:32`
- `api/regression/_linear.py:69`
- `api/artifacts/_prediction.py:53`, including `prediction_kinds` at `:164` and `:199`

Put the rule in one helper in `api/artifacts/_prediction.py`, since `api.regression` already depends on that package. Removing `linear` changes persisted `prediction_kinds`, so apply the schema rule from the handoff guide.

## Non-goals

Mapping and frame inputs belong to later sub-tasks.

Follow the [handoff guide](../../handoff.md). Measurements are in [evidence](../../evidence.md).

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

- [Accept mappings and scalar broadcasting on regression inputs](../mapping-inputs/README.md)

<!-- insitu:end -->
