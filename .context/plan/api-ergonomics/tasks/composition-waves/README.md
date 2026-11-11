---
type: task
title: Apply the composition-review waves
description: "Apply the 2026-11-06 composition-review findings in waves: permutation designs and shared WX selection first, then warrants, model description and bootstrap plotting."
links: [".context/plan/api-ergonomics/tasks/research-workflow/README.md"]
status: in-progress
---

# Apply the composition-review waves

## Goal

The 2026-11-06 review stress-tested STATADIC's composition across spatial, panel,
GLM, survival, ANOVA, PERMANOVA, DISCO and classical MDS. Its correctness
findings landed in phases 8 and 9: schema checks, supplied-schedule
provenance, panel roles in `prepare()`, SE kept visible when the interval
fails, and inference-time columns. This task orders what remains. Each wave
is one sub-task, done in the order below unless the owner reorders it.

## Acceptance

- [x] Wave 1: [exchangeability](tasks/exchangeability/README.md) and the shared
  WX selector from [wx-term](tasks/wx-term/README.md).
- [ ] Wave 2: [geometry-warrants](tasks/geometry-warrants/README.md).
- [ ] Wave 3: [model-description](tasks/model-description/README.md).
- [ ] Wave 4: the WX design term in [wx-term](tasks/wx-term/README.md).
- [ ] Anywhere: [bootstrap-intervals](tasks/bootstrap-intervals/README.md).
- [ ] Pass `just check` after each wave and report any check that could not run.

## Order

| Wave | Sub-task | Why this position |
| --- | --- | --- |
| 1 | exchangeability | The only remaining item that gave a wrong p-value for a common design. |
| 1 | wx-term (selector only) | Removes a duplicated contract; no design decision needed. |
| 2 | geometry-warrants | Sound today but over-restrictive; the extension guide must document the final warrant API. |
| 3 | model-description | Ergonomics; the target table should describe the survival family once it moves. |
| 4 | wx-term (design term) | Depends on graph-roster prediction and an artifact codec; feeds the target table's required inputs. |
| any | bootstrap-intervals | Small and independent. |

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| backlog | [Make bootstrap intervals plottable](tasks/bootstrap-intervals/README.md) | Let the shared coefficient plot label and draw percentile-bootstrap intervals, not only Wald intervals. |
| backlog | [Separate D and squared-D warrants, then publish the extension guide](tasks/geometry-warrants/README.md) | Let a geometry warrant negative type of D without a Hilbertian (squared-D) warrant, so DISCO accepts L1-like geometries; then write the extension guide around _L1OnMeans. |
| backlog | [Put the survival family on the model, then add prediction-target tables](tasks/model-description/README.md) | Make the implied statistical model inspectable: the AFT family moves from SurvivalML to the model, then results describe their prediction targets and required inputs. |
| backlog | [Share WX lag selection, then decide the WX design term](tasks/wx-term/README.md) | Share named WX lag selection across spatial hosts (done), then decide whether exogenous WX becomes a prepared, graph-aware design term usable by ordinary GLMs. |

`4 open; 1 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
