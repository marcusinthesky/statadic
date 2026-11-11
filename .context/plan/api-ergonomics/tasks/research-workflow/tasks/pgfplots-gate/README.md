---
type: task
title: Compile PGFplots output in the pinned TeX gate
description: "Prime pgfplots in the pinned TeX cache and compile PlotSpec output in tests, so native plots cannot silently break."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Compile PGFplots output in the pinned TeX gate

## Goal

`PlotSpec.to_pgfplots()` is a promised output, but no test compiles it and the cached bundle lacks `pgfplots.sty`. Add it to the existing TeX gate.

## Acceptance

- [x] `just python::docs::setup` primes `pgfplots` from the pinned bundle and retries through `infra/nix/retry-tex-setup.sh`.
- [x] A test compiles line, scatter and labelled `coefficient_interval` plots with a labelled reference line, using `only_cached=True`, with no Overfull box.
- [x] `docs/reference/composition.md` states the required preamble and no longer mentions a manuscript PGF transport, which does not exist in the repository.
- [x] The `src/latex` manuscript workspace is not changed by this sub-task. It replaced `main.tex` with per-project builds and compiles its own PGFplots fixture, `tools/tests/fixtures/pgfplots`, from a vendored bundle.
- [x] `just python::docs::check` and `just check` pass.

## Specification

- Extend `apps/docs/scripts/compile_report.py` with a figure compile and add `apps/docs/templates/figure.tex` containing `\usepackage{pgfplots}` and `\pgfplotsset{compat=1.18}`. Its `main` must prime both templates.
- Add the test to `apps/docs/tests/test_reporting_workflow.py`.
- Use `width_cm=10` in fixtures, because a 12 cm figure with a legend overflowed by 9.7pt. Do not weaken the Overfull check.
- CI runs `just setup` at `.github/workflows/ci.yml:40`, so priming through setup reaches hosted runs. Check the cache key there still restores correctly.

## Non-goals

No plotting features, Matplotlib PGF or `xelatex`.

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

- [Extend PlotSpec with ribbons, overlays and trusted labels](../plotspec-extension/README.md)

<!-- insitu:end -->
