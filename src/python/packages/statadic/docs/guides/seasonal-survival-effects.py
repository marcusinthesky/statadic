# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.2", "matplotlib>=3.8", "statadic @ git+https://github.com/marcusinthesky/statadic.git#subdirectory=src/python/packages/statadic"]
#
# [tool.marimo.runtime]
# reactive_tests = false
# ///
"""A reproducible STATADIC example with simulated observations."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.data import Observations, PanelLayout, MeasureColumn
    from statadic.api.design import Intercept, Linear, CyclicSpline
    from statadic.api.inference import ModelBased
    from statadic.api.regression import Regression
    from statadic.api.representation import Energy
    from statadic.api.spatial.graphs import PanelData, SpatialLag, SpatialX, DistanceGraph, ProjectionRef, RBFWeights
    from statadic.api.spatial.learning import GraphML, LearnLinear
    from statadic.api.spatial.resampling import TimeBlockPairs
    from statadic.api.survival import RightCensored, SpatialSurvivalData
    from statadic.estimation.survival import SurvivalML, SpatialSurvivalML
    from statadic.model.survival import LogNormalAFT
    from statadic.operators.spatial import prepare_spatial_weights
    from statadic.optimize.integration import IntegrationDesign
    from statadic.sample.packed import PackedClouds
    DARK = mo.app_meta().theme == "dark"
    INK = "#f2f2f2" if DARK else "#000000"
    BLUE = "#8fb8ff" if DARK else "#1d3f8f"
    RULE = "#555555" if DARK else "#d6d6d6"
    plt.rcParams.update({
        "figure.figsize": (6.4, 3.6), "figure.facecolor": "none",
        "axes.facecolor": "none", "savefig.format": "svg", "savefig.transparent": True,
        "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "text.color": INK, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "axes.prop_cycle": plt.cycler(color=[BLUE, "#8a8a8a", INK]),
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": RULE, "grid.linewidth": 0.6,
        "xtick.color": INK, "ytick.color": INK,
    })


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Add seasonal effects to spatial survival

    Use a cyclic spline in a spatial accelerated failure-time model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    When seasonal exposure varies among districts, encode phase on every graph node before fitting the observed spells. This example combines a cyclic basis, two-way effects and a fixed spatial lag in one model for the observed durations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    Four periods each contain six graph nodes, with one outcome missing and one right-censored among the observed spells. Phase is measured as a fraction of a year and varies within periods, preserving information about the seasonal cycle beyond the period effects.
    """)
    return


@app.cell
def _():
    data, streams, phase, event = seasonal_data()
    _fig, _ax = plt.subplots()
    _ax.hist(phase, bins=8)
    _ax.set_xlabel("Annual phase")
    _ax.set_ylabel("Graph nodes")
    _fig
    return data, event, streams


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Specify and fit the model

    A three-degree cyclic spline wraps at one year, joining the beginning and end of the seasonal cycle. Unit and period effects provide locations, and fixing rho makes the spatial dependence assumption explicit.
    """)
    return


@app.cell
def _(data, streams):
    model = Regression(RightCensored("duration", "event"), Linear("x") + CyclicSpline("phase", df=3, period=1.0), structure=data.panel, law=LogNormalAFT(), dependence=SpatialLag("neighbors"), effects="two-way", estimator=SpatialSurvivalML(rho_fixed=0.2))
    fit = model.estimate(SpatialSurvivalData(data, streams)).require_valid()
    mo.plain_text(fit.report().to_text())
    return (fit,)


@app.cell(hide_code=True)
def _(event, fit):
    mo.md(f"The fit uses **{event.size}** observed spells, including **{int((event == 0).sum())}** censored spell, "
          f"and retains **{len(fit.parameters)}** parameters. The full graph retains the missing node.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Predict marginal medians

    Predictions follow the graph order and incorporate propagation through the fixed spatial lag, so their positions still correspond to the prepared nodes.
    """)
    return


@app.cell
def _(data, fit):
    prediction = fit.predict(data, kind="median")
    _fig, _ax = plt.subplots()
    _ax.bar(np.arange(6), np.asarray(prediction.values[0]))
    _ax.set_xlabel("Unit in graph order")
    _ax.set_ylabel("Predicted median duration")
    _fig
    return (prediction,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Change the prediction horizon to see how survival varies with elapsed time, while keeping the fitted seasonal model and spatial graph fixed.
    """)
    return


@app.cell
def _():
    horizon = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="Elapsed time")
    horizon
    return (horizon,)


@app.cell
def _(data, fit, horizon):
    _pred = fit.predict(data, np.array(horizon.value), kind="survival")
    _fig, _ax = plt.subplots()
    _ax.bar(np.arange(6), np.asarray(_pred.values[0]))
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("Unit in graph order")
    _ax.set_ylabel("Marginal survival probability")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The fitted seasonal design retains predictors for every graph node and a replayable integration design for each period of the observed panel, preserving their order for later prediction.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Assumptions and evidence
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md('This small integration budget has not been refined. See the [assumptions and limits](../reference/survival.md#assumptions-and-limits) and [evidence](../reference/survival.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Spatial survival tutorial](../tutorials/spatial-panel-survival.py).
    - [Cyclic spline reference](../reference/composition.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def seasonal_data():
    """Construct full graph nodes with a missing and a censored outcome."""
    rng = np.random.default_rng(1027)
    n, periods = 6, 4
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    unit = np.tile(np.arange(n, dtype=np.int32), periods)
    year = np.repeat(np.arange(periods, dtype=np.int32), n)
    phase = rng.uniform(size=len(unit)).astype(dtype)
    x = rng.normal(size=len(unit)).astype(dtype)
    nodes = Observations({"unit": unit, "year": year, "phase": phase, "x": x})
    rows = rng.permutation(len(unit))[1:]
    event = np.ones(len(rows), dtype=np.int32)
    event[0] = 0
    outcomes = Observations({"unit": unit[rows], "year": year[rows], "duration": np.exp(0.2 * x[rows] + rng.normal(size=len(rows))).astype(dtype), "event": event}, row_ids=rows.tolist())
    graph = prepare_spatial_weights(np.roll(np.eye(n, dtype=dtype), 1, axis=1), unit_ids=np.arange(n, dtype=np.int32))
    streams = tuple(IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=8) for t in range(periods))
    return PanelData(nodes, PanelLayout("unit", "year"), {"neighbors": graph}, outcomes), streams, phase, event


@app.cell(hide_code=True)
def _(event, fit, prediction):
    def test_seasonal_fit_predicts_all_graph_nodes():
        assert fit.valid
        assert len(prediction.values[0]) == 6
        assert int((event == 0).sum()) == 1
        assert np.all(np.asarray(prediction.values[0]) > 0)

    return


if __name__ == "__main__":
    app.run()
