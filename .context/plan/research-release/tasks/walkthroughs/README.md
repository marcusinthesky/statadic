---
type: task
title: Prove the OLS and Poisson research journeys on the installed wheel
description: "Two canonical walkthroughs, from a dataframe to a reloaded predictor, executed in the docs lanes and against the built wheel."
links: [".context/plan/research-release/handoff.md", ".context/plan/research-release/evidence.md"]
status: done
---

# Prove the OLS and Poisson research journeys on the installed wheel

## Goal

A researcher new to STATADIC can start from a Polars or Arrow table, fit a model, inspect its validity, choose inference, produce a paper-ready table and figure, save a predictor, reload it, and reproduce the results. They should need no custom serialization code, no hand-built graphics coordinates, and no guessing about statistical validity. Test this as one journey per model, on the installed wheel.

## Acceptance

- [x] `examples/first_regression.py` is the OLS/HC3 walkthrough and `examples/poisson_exposure.py` is the Poisson walkthrough with exposure, both following the journey below.
- [x] Neither script imports `Observations`, builds `PlotSpec` coordinates by hand, converts a frame to arrays, or writes custom serialization.
- [x] `statadic.api.plots.residual_plot` exists with mirrored tests, and the public-API updates listed below are done.
- [x] Both scripts run in both precision lanes of `apps/docs/scripts/check_examples.py`, and the tutorials build.
- [x] `just python::statadic::test-artifacts` runs both scripts from the extracted sdist against the installed wheel with the `data` extra, in both precision lanes.
- [x] `verification.md` records any friction met while writing the scripts, especially with `prediction_line`, `coefficient_plot` or `residual_plot`.
- [x] Focused tests, `just python::architecture`, lint, typecheck and `just check` pass.

## Journey

Both scripts take `--output DIR` and default to a temporary directory, so a run never writes into the checkout.

1. Generate seeded synthetic data with `jax.random` and place it in a Polars `DataFrame` for OLS, or a `pyarrow.Table` for Poisson.
2. Fit with `model.fit(frame).require_valid()`, then print `fit.report().to_text()`.
3. Infer with `fit.infer(covariance=HC("HC3"))` for OLS or `HC("HC1")` for Poisson, then `.report(reference="normal")`. Print it and write `report.to_latex()`.
4. Make figures:
   - OLS: `residual_plot(fit)` and `report.coefficient_plot()`.
   - Poisson: `prediction_line(fit, "load", grid, kind="rate")`, needing no exposure.
5. Write each figure with `to_pgfplots(width_cm=10)`.
6. Save with `fit.save_predictor(DIR / "predictor")`. Check that `load_predictor(...).predict(new)` equals `fit.predict(new)` on new rows supplied as a mapping.
7. Save with `save_analysis(DIR / "analysis", replace(report, plots=(...)))`. Check that `load_analysis` re-renders identical LaTeX.

Keep the `# tutorial-start` and `# tutorial-end` markers, because `docs/tutorials/first-regression.md` includes the OLS script by `literalinclude`.

## Specification

**`residual_plot(fit) -> PlotSpec`** in `api/plots.py`:

- It is a scatter of retained `fit.core.fitted_values` against response residuals, the retained response minus the fitted values.
- It accepts only `LinearResult`, which covers OLS and WLS. Other families raise `TypeError` naming the supported type.
- It calls `fit.require_valid()` first.
- It adds a zero reference line, axis labels and metadata naming the residual type.
- It does no standardization, studentization or smoothing, because those are new statistics.

Public-API updates:

- `tests/api/test_plots.py`;
- `docs/reference/composition.md`, next to `prediction_line`;
- the CHANGELOG `[Unreleased]` section;
- the host-wrapper entry in `apps/studies/catalogue.json`;
- `just lean::inventory-refresh`.

**Docs:**

- Update `docs/tutorials/first-regression.md`.
- Add `docs/tutorials/poisson-exposure.md` and list it in the toctree of `docs/start/index.md`.
- Register `poisson_exposure.py` in `apps/docs/scripts/check_examples.py`.

**Wheel lane:** in `scripts/check_distribution.py`, add a function next to `_check_wheel` that:

1. creates a second venv;
2. installs `"<wheel>[data]"`;
3. copies both examples from the extracted sdist, never the checkout, into an empty directory;
4. runs each with `python -I` under `JAX_ENABLE_X64=0` and `=1`, with `--output` in a temporary directory.

The lane must assert that `statadic` imports from that venv. Running it for the direct wheel is enough.

## Non-goals

- No `fit.plot` accessor; see [plot-accessor](../plot-accessor/README.md).
- No prediction bands.
- No changes to `Observations`.
- No new estimator or inference options.

Follow the [handoff guide](../../handoff.md). Findings are in [evidence](../../evidence.md).

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

- [Build the synthetic OLS and HC3 paper](../../../publishing-companion/tasks/paper/README.md)
- [Defer a fitted-result plot accessor](../plot-accessor/README.md)

<!-- insitu:end -->
