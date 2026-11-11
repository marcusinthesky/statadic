---
type: task
title: Update templates, examples and docs for the new exports
description: Move examples, package docs and the publishing example to the new table format and save verbs, and record the change.
links:
  - .context/plan/api-cohesion/tasks/report-review/audit.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/latex-tables/specification.md
status: done
---

# Update templates, examples and docs for the new exports

## Goal

After `latex-tables`, a report table needs `siunitx` in the preamble, and
`save_latex` and `save_pgfplots` replace hand-written file writes. Row P07 of the
[dispositions](../../../../proposal-dispositions.md) accepts bringing every
consumer and page in line. Start after `latex-tables` is done.

Source under `src/python/packages/statadic/src/` and the files of `latex-tables`,
`report-content` and `pgfplots-output` are not edited here.

## Acceptance

- [x] The publishing example's manuscript loads `siunitx` and its bundle builds and verifies offline.
- [x] `src/latex/tools/tests/fixtures/publishing/main.tex` typesets the same constructs as the example, so a later bundle refresh keeps its closure.
- [x] Shipped examples and the publishing analysis write tables with `report.save_latex(...)` and figures with `plot.save_pgfplots(...)` once that method exists.
- [x] The reporting guide and reference page state the required preamble (`booktabs`, `siunitx`, and `pgfplots` for figures) and document `digits`, `decimals`, `labels`, `notes` and the save verbs.
- [x] No page or example names `summary()` or `report(level=)`.
- [x] `CHANGELOG.md` records the table format, the save verbs, the removed alias and the renamed keyword, with no compatibility framing.
- [x] The docs size budget, example checks and docs reporting tests pass.

## Specification

- Files: `src/python/packages/statadic/examples/`, `src/python/packages/statadic/docs/`
  (`guides/reporting.md`, `reference/reporting.md`, `tutorials/`,
  `guides/graph-resampling.md`), `src/python/packages/statadic/CHANGELOG.md`,
  `src/python/packages/statadic-publishing/examples/ols-hc3/` and the fixture above.
- The vendored bundle holds only fonts that current documents use. If the
  publishing manuscript cannot compile offline with the new table, give it the
  docs template's preamble (`\usepackage[T1]{fontenc}`, `\usepackage{lmodern}`),
  which the audit compiled. Do not run `just latex::bundle-refresh`.
- `publishing.toml` pins sha256 values for its inputs. Repin any pinned file this
  task changes.
- If `save_pgfplots` does not exist yet, keep the figure write as it is and say so
  in the handback.
- Tracked files under `src/latex/projects/*/src/generated/` and the paper's
  listings are not regenerated: generation records the tree state and needs a
  commit first.

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
