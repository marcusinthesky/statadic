# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.25.2",
#     "matplotlib>=3.8",
#     "statadic @ git+https://github.com/marcusinthesky/statadic.git#subdirectory=src/python/packages/statadic",
# ]
#
# [tool.marimo.runtime]
# reactive_tests = false
# ///
"""Change the geometry used to compare distributions."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.representation import Energy
    from statadic.sample.packed import PackedClouds
    DARK = mo.app_meta().theme == "dark"
    INK = "#f2f2f2" if DARK else "#000000"
    BLUE = "#9bbaff" if DARK else "#1d3f8f"
    RULE = "#555555" if DARK else "#d6d6d6"
    plt.rcParams.update({
        "figure.figsize": (6.4, 3.6), "figure.facecolor": "none",
        "axes.facecolor": "none", "savefig.format": "svg", "savefig.transparent": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK,
        "axes.prop_cycle": plt.cycler(color=[BLUE, "#8a8a8a", INK]),
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": RULE, "grid.linewidth": 0.6,
    })


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Change geometry

    Compare the same two empirical measures with Euclidean and angular ground distances.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A ground distance determines which differences enter an energy comparison, and the right choice depends on the scientific question. Changing that ground while preserving the atoms and row identities lets you examine the consequences of the choice.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Prepare both geometries

    The second measure is a positive rescaling of the first, so its atoms lie in the same directions but at different distances from the origin. Both preparations receive the same measure carrier and the same ordered identities, which lets the resulting matrices be compared row by row.
    """)
    return


@app.cell
def _():
    measures = make_measures()
    ids = ("original", "scaled")
    euclidean = Energy(ground="euclidean").prepare(measures, row_ids=ids).pairwise()
    angular = Energy(ground="angular").prepare(measures, row_ids=ids).pairwise()
    mo.plain_text(f"Euclidean: {euclidean.values}\nAngular: {angular.values}")
    return angular, euclidean


@app.cell(hide_code=True)
def _(angular, euclidean):
    mo.md(f"The rooted Euclidean distance is **{float(euclidean.values[0, 1]):.2f}** and the angular distance is **{float(angular.values[0, 1]):.2f}**. Angular ground preserves the direction of each vector while discarding changes in its positive scale.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how each distance changes when the second measure is rescaled, then
    move the control to compare the results. At a scale of one, the second
    measure contains exactly the same atoms as the first.
    """)
    return


@app.cell
def _():
    scale = mo.ui.slider(1.0, 3.0, step=0.25, value=3.0, label="Scale of the second measure")
    scale
    return (scale,)


@app.cell
def _(scale):
    _measures = make_measures(scale.value)
    _ids = ("original", "scaled")
    _euclidean = Energy(ground="euclidean").prepare(_measures, row_ids=_ids).pairwise()
    _angular = Energy(ground="angular").prepare(_measures, row_ids=_ids).pairwise()
    mo.md(f"Euclidean: **{float(_euclidean.values[0, 1]):.3f}** · Angular: **{float(_angular.values[0, 1]):.3f}**")
    return


@app.cell(hide_code=True)
def _(scale):
    explore_figure(scale)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The row identities remain aligned across the two matrices, even though the distances answer different questions about the same measures. Choose Euclidean ground when differences in the magnitude of the observations belong in the comparison of their measures.
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
    mo.callout(mo.md('Angular ground requires nonzero vectors. See the [assumptions and limits](../reference/energy.md#assumptions-and-limits) and [evidence](../reference/energy.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Energy and scale](../tutorials/energy-and-scale.py).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def explore_figure(scale):
    _measures = make_measures(scale.value)
    _ids = ("original", "scaled")
    _e = Energy(ground="euclidean").prepare(_measures, row_ids=_ids).pairwise()
    _a = Energy(ground="angular").prepare(_measures, row_ids=_ids).pairwise()
    _fig, _ax = plt.subplots()
    _ax.bar(["Euclidean", "Angular"], [float(_e.values[0, 1]), float(_a.values[0, 1])])
    _ax.set_ylabel("Rooted energy distance")
    return _fig


@app.function(hide_code=True)
def make_measures(scale: float = 3.0):
    """Construct a pair of empirical measures with the same directions."""
    x = jnp.array([[1.0], [2.0]])
    return PackedClouds.from_clouds([x, scale * x])


@app.cell(hide_code=True)
def _(angular, euclidean):
    def test_result():
        assert euclidean.row_ids == angular.row_ids == ("original", "scaled")
        assert abs(float(euclidean.values[0, 1]) - 2.0) < 1e-5
        assert abs(float(angular.values[0, 1])) < 1e-5

    return


if __name__ == "__main__":
    app.run()
