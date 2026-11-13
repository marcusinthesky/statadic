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
    # Compose survival, panel identity and spatial dependence

    Fit repeated spells and compare a fixed spatial dependence assumption.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    When a district contributes one observed duration in each year, a spatial graph can describe relationships with nearby districts while preserving the identity of each observation. We will fit a lognormal accelerated failure-time model and inspect predictions with that graph, the ordering of its nodes and the unit and period identities stated explicitly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    Read the [survival reference](../reference/survival.md) for censoring and accelerated failure-time models.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    Six units contribute a spell in each of three periods, producing 18 observations with a shared unit identity across time and a matching graph order. One predetermined observation limit censors the first spell, while the remaining events are observed.
    """)
    return


@app.cell
def _():
    spells, panel_data, streams, duration, event = survival_data()
    _fig, _ax = plt.subplots()
    _ax.scatter(np.tile(np.arange(6), 3), duration.ravel(), c=event.ravel(), cmap="viridis")
    _ax.set_xlabel("Unit")
    _ax.set_ylabel("Observed duration")
    _preview = mo.ui.table([{"unit": int(_i), "duration": float(duration.ravel()[_i]), "event": int(event.ravel()[_i])} for _i in range(5)])
    mo.vstack([_preview, _fig])
    return event, panel_data, spells, streams


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit pooled and period effects

    An intercept supplies the pooled location, allowing the first fit to summarize the spells without estimating a separate location for each period. The second specification uses period fixed effects, so its named predictor design omits the intercept.
    """)
    return


@app.cell
def _(panel_data, spells):
    response = RightCensored("time", "event")
    pooled = Regression(response, Intercept() + Linear("x"), law=LogNormalAFT(), estimator=SurvivalML(family=LogNormalAFT())).estimate(spells).require_valid()
    time_fit = Regression(response, Linear("x"), law=LogNormalAFT(), estimator=SurvivalML(family=LogNormalAFT()), structure=panel_data.panel, effects="time").estimate(spells).require_valid()
    mo.plain_text(time_fit.report().to_text())
    return pooled, response, time_fit


@app.cell(hide_code=True)
def _(event, pooled, time_fit):
    mo.md(f"The pooled fit has **{len(pooled.parameters)}** parameters for a panel of "
          f"**{event.size}** spells with **{int(event.sum())}** observed events. "
          f"The period-effects model has **{len(time_fit.parameters)}** fitted parameters, including log scale.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Add a spatial outcome lag

    Each period uses the same row-normalized complete graph in unit order, and the simulated data have no spatial dependence. Compare fixed rho values of zero and 0.2 to see how the analysis responds to a specified dependence strength.
    """)
    return


@app.cell
def _(panel_data, response, streams):
    model = Regression(response, Linear("x"), structure=panel_data.panel, law=LogNormalAFT(), dependence=SpatialLag("neighbors"), estimator=SpatialSurvivalML(rho_fixed=0.2), effects="time")
    baseline_model = Regression(response, Linear("x"), structure=panel_data.panel, law=LogNormalAFT(), dependence=SpatialLag("neighbors"), estimator=SpatialSurvivalML(rho_fixed=0.0), effects="time")
    baseline = baseline_model.estimate(SpatialSurvivalData(panel_data, streams)).require_valid()
    fit = model.estimate(SpatialSurvivalData(panel_data, streams)).require_valid()
    inference = fit.infer(covariance=ModelBased())
    prediction = fit.predict(panel_data, np.array(1.0), kind="survival")
    _fig, _ax = plt.subplots()
    _baseline_prediction = baseline.predict(panel_data, np.array(1.0), kind="survival")
    _ax.bar(np.arange(6) - 0.2, np.asarray(_baseline_prediction.values[0]), 0.4, label="rho = 0")
    _ax.bar(np.arange(6) + 0.2, np.asarray(prediction.values[0]), 0.4, label="rho = 0.2")
    _ax.legend()
    _ax.set_xlabel("Unit in graph order")
    _ax.set_ylabel("Survival probability at time 1")
    _fig
    return fit, inference, prediction


@app.cell(hide_code=True)
def _(inference, prediction):
    mo.md(f"For the first unit, the marginal survival probability at time one is **{float(prediction.values[0][0]):.3f}**, "
          f"and availability of model-based covariance is **{bool(inference.uncertainty.valid)}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Change the evaluation time to see how predicted survival varies across units, while the fitted graph and model remain fixed.
    """)
    return


@app.cell
def _():
    horizon = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="Elapsed time")
    horizon
    return (horizon,)


@app.cell
def _(fit, horizon, panel_data):
    _prediction = fit.predict(panel_data, np.array(horizon.value), kind="survival")
    _fig, _ax = plt.subplots()
    _ax.bar(np.arange(6), np.asarray(_prediction.values[0]))
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("Unit in graph order")
    _ax.set_ylabel("Marginal survival probability")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - Period effects and spatial dependence specify distinct features of a repeated-spell model.
    - The fixed rho is an assumption, and predictions are marginal over neighboring outcomes.
    - Integration draws and the order of nodes are part of the model input, so they travel with the observed panel.
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
    mo.callout(mo.md('This integration budget is illustrative and unrefined. See the [assumptions and limits](../reference/survival.md#assumptions-and-limits) and [evidence](../reference/survival.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Add seasonal effects](../guides/seasonal-survival-effects.py).
    - [Review spatial panels](../reference/spatial-panels.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def survival_data():
    """Construct ordered nodes, one censored spell, and replayable draws."""
    rng = np.random.default_rng(241)
    n, periods = 6, 3
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    x = rng.normal(size=(periods, n)).astype(dtype)
    latent = np.exp(0.4 + 0.3 * x + 0.5 * rng.normal(size=(periods, n)))
    limits = np.full_like(latent, np.inf)
    limits[0, 0] = 1.0
    duration = np.minimum(latent, limits).astype(dtype)
    event = (latent <= limits).astype(np.int32)
    unit = np.tile(np.arange(n, dtype=np.int32), periods)
    period = np.repeat(np.arange(periods, dtype=np.int32), n)
    nodes = Observations({"x": x.ravel(), "unit": unit, "period": period})
    spells = Observations({"time": duration.ravel(), "event": event.ravel(), "unit": unit, "period": period, "x": x.ravel()})
    layout = PanelLayout("unit", "period")
    weights = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    graphs = tuple((t, prepare_spatial_weights(weights, unit_ids=np.arange(n))) for t in range(periods))
    streams = tuple(IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=16) for t in range(periods))
    return spells, PanelData(nodes, layout, {"neighbors": graphs}, spells), streams, duration, event


@app.cell(hide_code=True)
def _(event, fit, prediction):
    def test_spatial_prediction_is_a_probability():
        assert int(event.sum()) == 17
        assert fit.valid
        assert np.all((np.asarray(prediction.values[0]) >= 0) & (np.asarray(prediction.values[0]) <= 1))

    return


if __name__ == "__main__":
    app.run()
