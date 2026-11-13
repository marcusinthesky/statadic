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
    # Learn a graph from unit distributions

    Fit a shared projection of article coordinates together with a graph regression.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    Each firm owns a distribution of article coordinates, while the graph has one row per firm and a shared projection maps every article before distances between firms are calculated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    Seven firms have two simulated article vectors each, together with five dates of returns and firm-level predictors. The returns are generated from a fixed projected graph, giving the learning procedure a known synthetic structure to fit.
    """)
    return


@app.cell
def _():
    data, semantic, x, y = learning_data()
    _fig, _ax = plt.subplots()
    _ax.scatter(x.ravel(), y.ravel(), s=20)
    _ax.set_xlabel("Firm predictor")
    _ax.set_ylabel("Return")
    _fig
    return data, semantic


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit a shared linear projection

    Energy compares the projected empirical distributions, and RBF weights convert their costs to a graph with one row per firm and a column for each potential neighbor. The likelihood estimates the shared projection together with the regression coefficients.
    """)
    return


@app.cell
def _(data, semantic):
    model = Regression("y", Intercept() + Linear("x") + SpatialX("x", "information"), estimator=GraphML(projections={semantic: LearnLinear(1, initial=jnp.array([[0.8], [0.15]]))}))
    fit = model.estimate(data).require_valid()
    mo.plain_text(fit.report().to_text())
    return (fit,)


@app.cell(hide_code=True)
def _(fit):
    projection = np.asarray(fit.graph_parameters["semantic"])
    mo.md(f"The learned map has shape **{projection.shape}** and norm **{np.linalg.norm(projection):.3f}**. "
          "Equivalent rotations or signs can represent the same geometry.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Inspect the graph

    Apply the fitted map to realize the graph, then inspect how its weights distribute each firm’s connections across the seven units. Learned geometry may be weakly identified, and fixed-graph inference excludes learning and selection. Repeat supervised learning within a justified resampling design when the full procedure is the target of inference.
    """)
    return


@app.cell
def _(data, fit):
    weights = np.asarray(data.graphs["information"].weights(fit.graph_parameters))
    _fig, _ax = plt.subplots()
    _image = _ax.imshow(weights, vmin=0)
    _ax.set_xlabel("Neighbor")
    _ax.set_ylabel("Firm")
    _fig.colorbar(_image, ax=_ax, label="Graph weight")
    _fig
    return (weights,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Choose a firm to see how its fitted graph row distributes weight across its neighbors, and compare that pattern with the other rows.
    """)
    return


@app.cell
def _():
    firm = mo.ui.slider(0, 6, value=0, label="Firm")
    firm
    return (firm,)


@app.cell
def _(firm, weights):
    _fig, _ax = plt.subplots()
    _ax.bar(np.arange(7), weights[firm.value])
    _ax.set_xlabel("Neighbor")
    _ax.set_ylabel("Graph weight")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The shared map is learned from the full regression objective, and the fitted graph remains conditional on this training sample. For a forecasting study, select any bandwidth on a later validation window and reserve another period for final evaluation.

    The linked neural example trains a 6 → 8 → 2 network and bounded spatial coefficient through energy costs and the Gaussian SAR profile likelihood, including its log determinant. It fixes the bandwidth and profiles regression coefficients and innovation variance. Its final regression freezes the learned graph, so the report excludes training uncertainty.
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
    mo.callout(mo.md('See the [assumptions and limits](../reference/spatial-panels.md#assumptions-and-limits) and [evidence](../reference/spatial-panels.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Refit whole-date blocks](resample-panel-graphs.py).
    - [Neural SAR guide](train-a-representation-through-sar.py), which trains a 6 → 8 → 2 Flax network through energy costs and the SAR profile likelihood.
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
def learning_data():
    """Generate article clouds and returns from a fixed latent graph."""
    rng = np.random.default_rng(16)
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    ids = tuple(range(7))
    clouds = PackedClouds.from_clouds([jnp.array([[float(i), (-1.0) ** i], [float(i) + 0.3, 0.5]]) for i in ids])
    ref = ProjectionRef("semantic")
    graph = DistanceGraph.from_measures(MeasureColumn(ids, clouds), geometry=Energy(), projection=ref, weighting=RBFWeights(1.3))
    weights = np.asarray(graph.weights({"semantic": jnp.array([[0.7], [0.3]])}))
    x = rng.normal(size=(5, len(ids))).astype(dtype)
    y = (0.3 + 0.7 * x + 2.1 * x @ weights.T + 0.04 * rng.normal(size=x.shape)).astype(dtype)
    rows = Observations({"x": x.ravel(), "y": y.ravel(), "date": np.repeat(np.arange(5, dtype=np.int32), len(ids)), "unit": np.tile(np.asarray(ids, dtype=np.int32), 5)})
    return PanelData(rows, PanelLayout("unit", "date"), {"information": graph}), ref, x, y


@app.cell(hide_code=True)
def _(fit, weights):
    def test_learned_graph_is_row_normalized():
        assert fit.valid
        assert np.allclose(weights.sum(axis=1), 1, atol=1e-5)
        assert np.allclose(np.diag(weights), 0)

    return


if __name__ == "__main__":
    app.run()
