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
"""Predict with a principal-coordinate regression."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import numpy as np
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.design import Intercept, Linear
    from statadic.api.inference import HAC, HC, Clustered, ModelBased
    from statadic.api.regression import Regression
    from statadic.api.data import Observations
    from statadic.api.ordination import LinearGram, Ordination, PrincipalCoordinates
    from statadic.estimation.iv import TwoStageLeastSquares

    DARK = mo.app_meta().theme == "dark"
    INK = "#eeeeee" if DARK else "#000000"
    BLUE = "#90b4ff" if DARK else "#1d3f8f"
    RULE = "#555555" if DARK else "#d6d6d6"
    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 3.2),
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.format": "svg",
            "savefig.transparent": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "legend.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fit principal-component regression

    Reduce correlated predictors to coordinates learned from training rows, then use them in a regression.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    The simulated response depends on two predictors, and the final two observations are held out to demonstrate prediction on new rows. The coordinate transformation is learned only from the training sample and then applied unchanged to the held-out observations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    The [first regression](../start/first-regression.py) introduces the fitted-result workflow.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Prepare the observations

    Create the sample and set aside the last two observations, so the coordinate map is learned from the training rows alone.
    """)
    return


@app.cell
def _():
    data = make_data()
    training = data.take(tuple(range(10)))
    held_out = data.take((10, 11))
    return held_out, training


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Fit coordinates within the design

    Place the principal coordinate in the regression design, where it is fitted from the training predictors before the coefficients are estimated.
    """)
    return


@app.cell
def _(training):
    components = Ordination(LinearGram(features=("x", "z"), scale=True), PrincipalCoordinates(dim=1), name="principal")
    fit = Regression("y", Intercept() + components).estimate(training).require_valid()
    mo.Html(fit.report().to_html())
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Predict held-out observations

    Prediction applies the same scaling and principal coordinate learned from training to each held-out row, rather than fitting a new transformation to those observations.
    """)
    return


@app.cell
def _(fit, held_out):
    prediction = fit.predict(held_out)
    mo.md(f"Predicted means: {prediction.tolist()}")
    return (prediction,)


@app.cell(hide_code=True)
def _(prediction):
    mo.md(f"The first held-out prediction is {float(prediction[0]):.2f}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Move the first predictor while the second stays fixed, and compare the selected prediction with the fitted response across the range.
    """)
    return


@app.cell
def _():
    x_value = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.0, label="First predictor")
    x_value
    return (x_value,)


@app.cell
def _(fit, x_value):
    mo.md(f"Predicted mean: {float(fit.predict({"x": [x_value.value], "z": [0.0]})[0]):.2f}")
    return


@app.cell(hide_code=True)
def _(fit, x_value):
    _x = jnp.linspace(-2.0, 2.0, 60)
    _y = np.asarray(fit.predict({"x": _x, "z": jnp.zeros_like(_x)}))
    _selected = float(fit.predict({"x": [x_value.value], "z": [0.0]})[0])
    _fig, _ax = plt.subplots()
    _ax.plot(_x, _y)
    _ax.scatter([x_value.value], [_selected])
    _ax.set(xlabel="First predictor", ylabel="Predicted response")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The fitted result keeps the principal-coordinate map learned from the training predictors, so it can transform held-out observations for prediction with the same scaling and orientation as the training data.
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
    mo.callout(mo.md('See the [assumptions and limits](../reference/ordination.md#assumptions-and-limits) and [evidence](../reference/ordination.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Regression](../reference/regression.md)
    - [First regression](../start/first-regression.py)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def make_data():
    """Generate the example observations deterministically."""
    x = jnp.linspace(-2.0, 2.0, 12)
    z = x**2 + 0.15 * jnp.sin(3 * x)
    return Observations({"x": x, "z": z, "y": 1 + 0.8 * x - 0.2 * z + 0.1 * jnp.cos(2 * x)})


@app.cell(hide_code=True)
def _(fit):
    def test_fit_is_valid():
        assert bool(fit.valid)

    return


if __name__ == "__main__":
    app.run()
