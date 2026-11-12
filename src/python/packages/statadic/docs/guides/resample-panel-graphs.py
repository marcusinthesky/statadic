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
    # Resample graph panels by date

    Refit a graph regression on circular blocks of complete dates.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A time block keeps the units, outcomes and graph for each date together when a resampled panel is assembled. Use this approach to examine variation across refits when a block design is appropriate for dependence in the temporal process. Choose a block length that reflects a defensible dependence horizon.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    The seeded panel contains four units across five dates, with a changing graph that aligns to the same ordered units and travels with each sampled date. Each response depends on its own predictor and the graph lag of that predictor.
    """)
    return


@app.cell
def _():
    data, x, y = graph_data()
    _fig, _ax = plt.subplots()
    _ax.plot(np.arange(80, 85), y)
    _ax.set_xlabel("Date")
    _ax.set_ylabel("Response by unit")
    _fig
    return (data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit the original graph regression

    The unit effects absorb persistent differences between units, while the two predictor coefficients describe the modeled relationship within this panel. For each date, the graph-lagged predictor is rebuilt from the corresponding aligned graph.
    """)
    return


@app.cell
def _(data):
    model = Regression("y", Linear("x") + SpatialX("x", "w"), effects="unit", estimator=GraphML())
    fit = model.estimate(data).require_valid()
    mo.plain_text(fit.report().to_text())
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Refit blocks of complete dates

    Draw five circular block samples of length two, keeping each date together with its units and graph. A 50% percentile interval leaves at least one draw in each tail of this small demonstration.
    """)
    return


@app.cell
def _(fit):
    boot = fit.infer(resampling=TimeBlockPairs(block_length=2, n_boot=5, seed=171))
    mo.plain_text(boot.report(confidence=0.5).to_text())
    return (boot,)


@app.cell(hide_code=True)
def _(boot):
    mo.md(f"The bootstrap completed **{boot.n_eff}** of **{len(boot.indices)}** requested refits. "
          f"The first sampled sequence of dates is **{[boot.periods[i] for i in boot.indices[0].tolist()]}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how a different seed changes the sampled dates while preserving adjacent pairs within each circular block. This control displays schedules without rerunning model fits.
    """)
    return


@app.cell
def _():
    seed = mo.ui.slider(171, 180, value=171, label="Resampling seed")
    seed
    return (seed,)


@app.cell
def _(seed):
    from statadic.operators.longrun import circular_block_bootstrap_indices
    _schedule = np.asarray(circular_block_bootstrap_indices(5, 2, 5, seed=seed.value))
    _fig, _ax = plt.subplots()
    _ax.imshow(_schedule, aspect="auto", vmin=0, vmax=4)
    _ax.set_xlabel("Position in resample")
    _ax.set_ylabel("Resample")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The result retains the sampled periods, refit counts and draws, so the reported uncertainty can be inspected against the resampling schedule and the failures observed during its execution, including the number of successful refits in each run.
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
    mo.callout(mo.md('Five refits do not establish coverage. See the [assumptions and limits](../reference/resampling.md#assumptions-and-limits) and [evidence](../reference/resampling.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Learn a graph](learn-panel-graphs.py).
    - [Spatial panel reference](../reference/spatial-panels.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def graph_data():
    """Generate a balanced panel with a changing graph on each complete date."""
    rng = np.random.default_rng(328)
    periods, n = 5, 4
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    x = rng.normal(size=(periods, n)).astype(dtype)
    weights = [np.roll(np.eye(n, dtype=dtype), 1 + t % 3, axis=1) for t in range(periods)]
    wx = np.einsum("tij,tj->ti", weights, x)
    y = (0.3 * x + 0.5 * wx + rng.normal(size=(periods, n))).astype(dtype)
    keys = {"unit": np.tile(np.arange(n, dtype=np.int32), periods), "date": np.repeat(np.arange(periods, dtype=np.int32) + 80, n)}
    nodes = Observations({"x": x.ravel(), **keys})
    outcomes = Observations({"y": y.ravel(), **keys})
    data = PanelData(nodes, PanelLayout("unit", "date"), {"w": tuple((t + 80, prepare_spatial_weights(w)) for t, w in enumerate(weights))}, outcomes)
    return data, x, y


@app.cell(hide_code=True)
def _(boot):
    def test_block_indices_preserve_circular_pairs():
        indices = np.asarray(boot.indices)
        assert boot.n_eff == 5
        assert np.all(indices[:, 1::2] == (indices[:, :-1:2] + 1) % 5)

    return


if __name__ == "__main__":
    app.run()
