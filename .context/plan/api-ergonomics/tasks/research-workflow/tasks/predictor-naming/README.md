---
type: task
title: Rename prediction artifact methods without aliases
description: "Rename prediction artifact methods and functions so their names say what they restore, with no aliases."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Rename prediction artifact methods without aliases

## Goal

`fit.save()` suggests a complete fitted model can be restored, but it writes a prediction-only artifact. Rename the surface so its names say what it restores.

## Acceptance

- [x] `save_predictor` replaces `save` on `LinearResult`, `PoissonResult`, `LogisticResult` and the PCA and PCoA results.
- [x] `statadic.api.artifacts` exposes `save_predictor`, `load_predictor`, `save_embedding_predictor` and `load_embedding_predictor`.
- [x] `save_prediction`, `load_prediction`, `save_embedding_prediction` and `load_embedding_prediction` are removed with no aliases.
- [x] On-disk content is unchanged, including `purpose="prediction"`, and this sub-task does not bump the schema.
- [x] Docs, tests, inventory and catalogue are updated.
- [x] `CHANGELOG.md` records a breaking change, and `just check` passes.

## Sites

- Methods: `api/regression/_linear.py:110`, `api/regression/_glm.py:115` and `:223`, `api/embedding.py:233` and `:546`
- Functions: `api/artifacts/__init__.py`, `api/artifacts/_prediction.py`, `api/artifacts/_embedding.py`
- Docs: `docs/reference/regression.md:18`, `docs/reference/composition.md:287`
- Tests: `tests/api/test_regression.py`, `test_embedding.py`, `test_regression_artifacts.py`, `test_artifacts.py`, `tests/integration/api/test_workflows.py`

`PredictionArtifact`, `PCAArtifact` and `PCoAArtifact` keep their names. To limit conflicts, run this after `prediction-roles` if both are in flight.

## Non-goals

No format change, Orbax support or new persistence capability.

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
