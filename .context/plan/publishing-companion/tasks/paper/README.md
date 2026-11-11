---
type: task
title: Build the synthetic OLS and HC3 paper
description: "A companion example project: synthetic data, an OLS/HC3 analysis producing snapshots, a table and a residual figure, and a short manuscript."
links: [".context/plan/release-experience/publishing-design.md", ".context/plan/publishing-companion/handoff.md", ".context/plan/research-release/tasks/walkthroughs/README.md"]
status: done
---

# Build the synthetic OLS and HC3 paper

## Goal

Provide the first consumer of the companion: a small, fully synthetic OLS paper with HC3 inference. It must be complete enough to build, verify, archive and rebuild. Researchers own the prose; the analysis owns every number.

Depends on [bundle](../bundle/README.md), and on [walkthroughs](../../../research-release/tasks/walkthroughs/README.md) for `residual_plot`.

## Acceptance

- [x] `src/python/packages/statadic-publishing/examples/ols-hc3/` contains:
  - `publishing.toml`;
  - `data/synthetic.csv`, together with the seeded `make_data.py` that wrote it;
  - `analysis.py`;
  - `manuscript/main.tex`.
- [x] Run as `python -I analysis.py --input <csv> --seed <n> --output <dir>`, the analysis writes exactly the declared outputs.
- [x] The manuscript compiles offline from the vendored bundle, and `check-log` passes.
- [x] A `reference` test compares the retained HC3 standard errors and coefficients with statsmodels.
- [x] `just python::publishing::check` passes.

## Specification

**Analysis**:

1. Read the CSV with Polars.
2. Fit `OLS(response="y", terms=Intercept() + Linear("x"))` with `.fit(frame).require_valid()`.
3. Save `fit.report()` with `save_analysis` as output `fit-snapshot`.
4. Run `fit.infer(covariance=HC("HC3")).report(reference="normal")`. Save it with its figures attached as `inference-snapshot`.
5. Write `table.tex` from `report.to_latex(digits=4)`.
6. Write `figure.tex` from `residual_plot(fit).to_pgfplots(width_cm=10)`.
7. Write a factual `methods.json` with `statadic_evidence.write_json`. It holds the model, terms, covariance, reference, number of observations, seed and `statadic` version, and no assumptions.

**Manuscript**:

- `article` class with `booktabs` and `pgfplots`, and `\pgfplotsset{compat=1.18}`.
- `\input{generated/table}` and `\input{generated/figure}`.
- A few sentences of researcher prose, citing `white-1980` and `mackinnon-white-1985` from `src/latex/refs.bib`.

**Specification file**: one claim about the slope's HC3 interval. Its statuses are explicit: `formal = "not-run"`, `source_fidelity = "not-run"`, `implementation = "pending"`, `qualification = "not-run"` and `human_review = "pending"`. The assumptions are researcher-written, such as independent observations and correct mean specification.

## Non-goals

- No real data.
- No multi-model table.
- No prediction bands.
- No journal template.

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

## Implementation

Follow the [handoff](../../handoff.md) and the [accepted design](../../../release-experience/publishing-design.md), including its 2026-11-06 amendment.
