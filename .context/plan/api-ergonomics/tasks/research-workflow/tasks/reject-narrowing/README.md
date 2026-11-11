---
type: task
title: Reject silent precision narrowing in Observations
description: "Make Observations refuse explicit 64-bit inputs when x64 is off, matching the Arrow and artifact boundaries."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Reject silent precision narrowing in Observations

## Goal

With JAX x64 off, `Observations` must refuse explicit 64-bit inputs instead of silently narrowing them. The owner accepted this on 2026-11-05. `ARCHITECTURE.md` already says numerical inputs retain their floating dtypes.

## Acceptance

- [x] With x64 off, NumPy float64, int64 and uint64 columns raise `ValueError` from `Observations.from_arrays` and the direct constructor. The message names the column and both fixes: enable x64 or cast explicitly.
- [x] With x64 off, float32, int32, JAX arrays, Python scalars and Python lists keep following JAX defaults.
- [x] With x64 on, float64 columns stay float64.
- [x] Arrow rejection (`api/tables.py:54`) is unchanged, and artifact checks still raise `ArtifactError`.
- [x] `docs/reference/composition.md` states one precision rule and shows `jax.config.update("jax_enable_x64", True)` and an explicit cast.
- [x] `CHANGELOG.md` records this as a breaking change.
- [x] `just python::docs::check` passes in both precision lanes, and `just check` passes.

## Specification

1. `from_arrays` and `_numeric` in `api/data.py` both call `jnp.asarray` with no dtype check.
2. Move the dtype logic from `check_array_precision` (`api/artifacts/_store.py:81`) into `statadic.api.data`. `api.artifacts` already depends on it. Let `_store` delegate and convert the failure to `ArtifactError`.
3. Run the check before the first `jnp.asarray` on both paths. Keep its recursion through lists and tuples.
4. An object whose `.dtype` is not a NumPy dtype, such as a Polars Series, must not be narrowed. A `TypeError` naming the column and `from_polars` is acceptable.
5. Add tests to `tests/api/test_data.py`.

Known consumers that may need explicit dtypes in the x64-off lane:

- `examples/change_estimator.py`
- `examples/spatial_panel_survival.py`
- `examples/distribution_workflow.py`
- `apps/docs/src/statadic_docs_labs/native.py`

## Non-goals

Do not change numerical kernels or other entry points such as `Mantel.fit` and `PackedClouds.from_clouds`. Record any narrowing you notice there in `verification.md`.

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
