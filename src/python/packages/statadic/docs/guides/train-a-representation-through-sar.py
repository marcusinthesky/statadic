# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.25.2",
#     "flax",
#     "optax",
#     "matplotlib>=3.8",
#     "statadic @ git+https://github.com/marcusinthesky/statadic.git#subdirectory=src/python/packages/statadic",
# ]
#
# [tool.marimo.runtime]
# reactive_tests = false
# ///
"""Learn a representation through a spatial likelihood."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.design import Intercept, Linear
    from statadic.api.regression import Regression
    from statadic.api.data import Observations

    DARK = mo.app_meta().theme == "dark"
    INK = "#eeeeee" if DARK else "#000000"
    BLUE = "#90b4ff" if DARK else "#1d3f8f"
    plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none", "savefig.format": "svg", "savefig.transparent": True, "font.family": "serif", "axes.prop_cycle": plt.cycler(color=[BLUE])})
    for _key in ("axes.edgecolor", "axes.labelcolor", "axes.titlecolor", "xtick.color", "ytick.color", "text.color", "legend.labelcolor"):
        plt.rcParams[_key] = INK


    import optax
    from flax import nnx
    from statadic.api.data import PanelLayout
    from statadic.api.representation import Energy
    from statadic.api.spatial.graphs import DistanceGraph, PanelData, RBFWeights, SpatialLag
    from statadic.estimation.spatial.graph_likelihood import graph_profile
    from statadic.operators.spatial.distance import rbf_weights
    from statadic.sample.packed import PackedClouds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Train a representation through SAR

    Learn a projection of distributions through a spatial autoregressive likelihood.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    Suppose each unit is described by a distribution of measurements, and nearby units may influence one another. This example learns a projection of the measurements through a Gaussian spatial panel likelihood, allowing the graph to respond to that projection.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    This guide uses Flax, Optax and the [spatial panel reference](../reference/spatial-panels.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Prepare distributions and observations

    The simulated sample has eight units and three atoms per unit, with outcomes and covariates observed across 24 periods. The generating graph is constructed independently from the network that we will train. This distinction lets us check a complete learning workflow without claiming that the simulated graph is known in an actual study.
    """)
    return


@app.cell
def _():
    clouds, x, y = simulate()
    mo.md(f"{x.shape[1]} units across {x.shape[0]} periods")
    return clouds, x, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Define the representation

    The network projects each atom into two dimensions, after which Euclidean energy compares the projected distributions. An RBF weighting turns those costs into a graph, and rho stays inside its stable range. The atom masses stay attached to their distributions throughout this transformation, so the comparison remains a comparison between measures.
    """)
    return


@app.class_definition
class NeuralSAR(nnx.Module):

    def __init__(self, seed=7):
        rngs = nnx.Rngs(seed)
        self.hidden = nnx.Linear(6, 8, rngs=rngs)
        self.projection = nnx.Linear(8, 2, rngs=rngs, use_bias=False)
        self.raw_rho = nnx.Param(jnp.asarray(0.2))

    def __call__(self, vectors):
        return jnp.tanh(self.projection(jnp.tanh(self.hidden(vectors))))

    @property
    def rho(self):
        return 0.95 * jnp.tanh(self.raw_rho[...])

    def costs(self, clouds):
        projected = clouds.with_coordinates(self(clouds.coordinates))
        energy = Energy(ground='euclidean').squared_cross(projected, projected)
        upper = jnp.triu(energy, k=1)
        return upper + upper.T

    def weights(self, clouds):
        return rbf_weights(self.costs(clouds), jnp.asarray(1.0))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Train through the likelihood

    The objective profiles out regression coefficients and innovation variance, while gradients pass through the energy costs, graph weights and spatial likelihood. Each optimizer step updates the projection and spatial coefficient together.
    """)
    return


@app.function(hide_code=True)
def profile(model, clouds, x, y):
    n_periods, n_units = x.shape
    lag = (model.weights(clouds)[None],) * n_periods
    error = (jnp.empty((0, n_units, n_units), dtype=x.dtype),) * n_periods
    design = tuple((jnp.column_stack((jnp.ones_like(row), row)) for row in x))
    return graph_profile(lag, error, design, tuple(y), model.rho[None])


@app.function
def loss(model, clouds, x, y):
    return -profile(model, clouds, x, y)[2] / y.size


@app.function
@nnx.jit
def train_step(model, optimizer, clouds, x, y):
    objective, gradients = nnx.value_and_grad(loss)(model, clouds, x, y)
    optimizer.update(model, gradients)
    return objective


@app.cell
def _(clouds, x, y):
    model = NeuralSAR()
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    initial = float(loss(model, clouds, x, y))
    for _ in range(120):
        train_step(model, optimizer, clouds, x, y)
    final = float(loss(model, clouds, x, y))
    mo.md(f"Loss per observation: {initial:.3f} → {final:.3f}")
    return final, initial, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4: Fit conditional on the learned graph

    After training, freeze the learned graph and fit a spatial regression to the observed panel, separating the conditional fit from the optimization that produced the graph. The final regression estimates its own spatial coefficient on that fixed graph, which need not equal the jointly trained coefficient. Compare both values: a difference can arise because the conditional refit optimizes its likelihood on the frozen graph, rather than updating the representation and graph together.
    """)
    return


@app.function
def conditional_fit(model, clouds, x, y):
    graph = DistanceGraph.from_costs(model.costs(clouds), unit_ids=tuple(range(x.shape[1])), weighting=RBFWeights(1.0))
    rows = Observations({'unit': jnp.tile(jnp.arange(x.shape[1]), x.shape[0]), 'period': jnp.repeat(jnp.arange(x.shape[0]), x.shape[1]), 'x': x.ravel(), 'y': y.ravel()})
    panel = PanelData(rows, PanelLayout('unit', 'period'), graphs={'learned': graph})
    return Regression('y', Intercept() + Linear('x'), dependence=SpatialLag('learned')).estimate(panel).require_valid()


@app.cell
def _(clouds, model, x, y):
    fit = conditional_fit(model, clouds, x, y)
    mo.vstack([
        mo.md(f"Trained rho: {float(model.rho):.3f}, conditional rho: {float(fit.parameters[-1]):.3f}"),
        mo.Html(fit.report().to_html()),
    ])
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Before selecting a unit, predict which neighboring weights will be largest in the learned graph. The chart displays the selected row of the same graph used for the conditional fit.
    """)
    return


@app.cell
def _():
    unit = mo.ui.slider(0, 7, value=0, label="Unit")
    unit
    return (unit,)


@app.cell
def _(clouds, model, unit):
    _weights = model.weights(clouds)[unit.value]
    _fig, _ax = plt.subplots()
    _ax.bar(range(8), _weights)
    _ax.set_xlabel("Neighboring unit")
    _ax.set_ylabel("Graph weight")
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The final report conditions on the learned graph, so its uncertainty excludes the additional variability introduced by training the representation.
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
    mo.callout(mo.md("See spatial [assumptions](../reference/spatial-panels.md#assumptions-and-limits) and [evidence](../reference/spatial-panels.md#evidence)."), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Spatial panels](../reference/spatial-panels.md)
    - [Energy](../reference/energy.md)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def simulate():
    rng = np.random.default_rng(1978)
    atoms = rng.normal(size=(8, 3, 6))
    z = np.tanh(atoms[..., :2])
    d = np.linalg.norm(z[:, None, :, None, :] - z[None, :, None, :, :], axis=-1).mean(axis=(-1, -2))
    w = np.exp(-(2 * d - d.diagonal()[:, None] - d.diagonal()[None, :]))
    np.fill_diagonal(w, 0)
    w /= w.sum(axis=1, keepdims=True)
    x = rng.normal(size=(24, 8))
    y = np.linalg.solve(np.eye(8) - 0.35 * w, (0.4 + 0.7 * x + rng.normal(scale=0.15, size=x.shape)).T).T
    return PackedClouds.from_array(jnp.asarray(atoms)), jnp.asarray(x), jnp.asarray(y)


@app.cell(hide_code=True)
def _(final, fit, initial):
    def test_training_and_conditional_fit():
        assert final < initial
        assert bool(fit.valid)

    return


if __name__ == "__main__":
    app.run()
