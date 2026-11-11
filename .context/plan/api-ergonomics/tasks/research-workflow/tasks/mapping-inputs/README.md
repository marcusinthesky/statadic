---
type: task
title: Accept mappings and scalar broadcasting on regression inputs
description: "Let regression fit and predict accept named mappings, with scalar broadcasting at prediction time."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/reject-narrowing/README.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/prediction-roles/README.md"]
status: done
---

# Accept mappings and scalar broadcasting on regression inputs

## Goal

Let regression `fit` and `predict` accept named mappings, so `Observations.from_arrays` and `jnp.ones_like(x)` stop being needed for everyday use. `Observations` stays the validated internal contract.

## Acceptance

- [x] `fit.predict({"load": grid}, kind="rate")` and `fit.predict({"load": grid, "hours": 8.0}, kind="mean")` match the explicit `Observations` results, live and reloaded.
- [x] `Regression.fit` and `IVRegression.fit` accept mappings and reject scalar columns.
- [x] At prediction, an unknown mapping key raises `ValueError` listing unknown and expected names. Length mismatches name the columns.
- [x] Categorical roles work with string and integer labels, and unseen levels raise the existing error.
- [x] An `Observations` input passes through unchanged. Panel, spatial and survival signatures are unchanged.
- [x] The narrowing rule applies to mapping values.
- [x] `tests/api/test_inputs.py`, Tach, inventory, catalogue, docs and `CHANGELOG.md` are updated, and `just check` passes.

Depends on `reject-narrowing` and `prediction-roles`.

## Specification

1. Add an `input_columns` property to `Term` and `FittedTerm` in `api/design.py`, returning `Mapping[str, Literal["numeric", "categorical", "measure"]]`. Implement it for every term and fitted term, plus `FittedDesign`. Conflicting roles raise.
2. Add `api/inputs.py` with `as_observations(data, roles, *, broadcast_scalars)`. It sits below `api.artifacts`, `api.base` and `api.regression` in the Tach graph.
3. Role sources:
   - Fit: the response, term roles and family roles (exposure, offset, trials, precision, endogenous, instruments, clusters).
   - Predict: design roles, family roles for the kind, and an optional exposure when supplied.
4. Broadcast scalars only when `broadcast_scalars=True`. Vector columns set the row count; all scalars means one row. Measures never broadcast.
5. At `fit`, keep extra mapping columns by value type so `Clustered("firm")` still works later. At `predict`, reject them.
6. Take the mapping as one positional argument. Do not add `**columns`, because a data column named `kind` must stay usable.
7. Change the `Regression` input type from `Observations` to the union and convert in `prepare`. `FitContext` keeps `Observations`. Annotations must satisfy the beartype import hook.

## Non-goals

No frame inputs, keyword columns, `from_formula` or grid objects.

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

- [Add coefficient-interval and prediction-line helpers](../plot-helpers/README.md)
- [Accept Arrow-compatible frames on regression inputs](../table-inputs/README.md)

<!-- insitu:end -->
