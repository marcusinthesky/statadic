---
type: task
title: Accept Arrow-compatible frames on regression inputs
description: "Let regression fit and predict accept Arrow-compatible frames through one protocol, with no pandas-specific code."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/mapping-inputs/README.md"]
status: done
---

# Accept Arrow-compatible frames on regression inputs

## Goal

Let regression `fit` and `predict` accept Polars frames, PyArrow tables and other Arrow-compatible frames through `__arrow_c_stream__`, with no pandas-specific code.

## Acceptance

- [x] A Polars frame and a PyArrow table work on `fit` and `predict`. A pandas frame works when pandas is installed, in a test gated on that optional dependency.
- [x] Only columns in declared roles are read, so an unrelated column with nulls or a wide dtype cannot block the call.
- [x] With x64 off, a default `Float64` frame raises the existing message, and docs show both fixes. With x64 on it works.
- [x] A single-chunk float32 column still shares its Arrow buffer, checked by comparing pointers.
- [x] `import statadic.api.tables` works with Polars absent, tested in a subprocess with `sys.modules["polars"] = None`.
- [x] A `LazyFrame` raises `TypeError` telling the caller to `collect()`. Nothing materializes implicitly.
- [x] A measure role raises `TypeError` pointing to `measures_from_arrow`.
- [x] `infer(covariance=Clustered(...))` on an undeclared column raises a clear error recommending `from_polars` with explicit roles.
- [x] Tests, Tach, docs and `CHANGELOG.md` are updated, and `just check` passes.

Depends on `mapping-inputs`.

## Specification

1. Detect frames in `as_observations` with `hasattr(data, "__arrow_c_stream__")`, then call `pa.table(data)` and the existing `from_arrow`.
2. Import PyArrow and `statadic.api.tables` inside the function. Never wrap imports in `try` or use `find_spec`. A missing PyArrow raises the normal `ModuleNotFoundError`, and docs name the `statadic[data]` extra.
3. Replace the module-level Polars import at `api/tables.py:14` with a small structural protocol exposing `to_arrow`. Keep `from_polars` and `measures_from_polars` as public names.
4. Frame columns outside declared roles are ignored at both `fit` and `predict`.
5. Keep `from_arrow` and `from_polars` for callers who want explicit roles, row identities or clusters.

## Non-goals

No change to `measures_from_arrow` performance, panel, spatial or survival inputs, automatic `collect()`, or `.to_jax()`.

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

- [Defer pandas ExtensionArray and xarray integration](../labelled-arrays/README.md)

<!-- insitu:end -->
