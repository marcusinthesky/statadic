---
type: task
title: Add require_valid to regression results
description: "Add a small chainable check that raises an informative error when a regression fit is unavailable."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Add require_valid to regression results

## Goal

Give host users one call that stops on an unavailable fit and explains why, without weakening the status-based behaviour numerical code relies on.

## Acceptance

- [x] `RegressionResult.require_valid()` returns `self` when valid and raises `InvalidFitError` otherwise.
- [x] `InvalidFitError` subclasses `ValueError` and is public in `statadic.api.base`.
- [x] The message names the model and includes the report's warnings, plus `solver_status`, `identified`, `finite_optimum` and `existence_status` when present.
- [x] A test covers a valid fit and a real invalid fit, such as a rank-deficient or separated design.
- [x] `valid`, fitting and numerical status behaviour are unchanged, with no fallback or refit.
- [x] Tach, inventory, catalogue, docs and `CHANGELOG.md` are updated, and `just check` passes.

## Specification

Implement it in `api/base.py` on `RegressionResult`, using `self.report()` for details. Add tests to `tests/api/test_base_lifecycle.py`. It is eager host code and must not be used under `jit`.

## Non-goals

No `explain()` method, automatic pseudoinverse, penalization or other change of statistical specification.

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

_No declared backlinks._

<!-- insitu:end -->
