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
"""Fit an instrumental-variables regression."""

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
    # Fit instrumental variables

    An instrument can identify a structural slope when a predictor is correlated with an unobserved shock.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    The simulated treatment depends on an assignment instrument and an unobserved shock that also affects the outcome, creating the endogeneity this example is designed to address. With a fixed random seed, the example lets you inspect the resulting instrumental-variables estimate and its uncertainty on the same set of observations.
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

    Create the sample with separate columns for the outcome, the endogenous treatment and the assignment used as its instrument.
    """)
    return


@app.cell
def _():
    data = make_data()
    return (data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Specify the endogenous predictor and its instrument

    Specify the endogenous treatment and assignment instrument alongside the regression terms, then estimate the model on the simulated observations.
    """)
    return


@app.cell
def _(data):
    fit = Regression("y", Intercept() + Linear("treatment"), estimator=TwoStageLeastSquares(), endogenous=("treatment",), instruments=("assignment",)).estimate(data).require_valid()
    mo.Html(fit.report().to_html())
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Request robust uncertainty

    The fitted result can attach HC0 covariance and a normal reference without rebuilding the structural model, keeping the uncertainty request separate from the coefficient estimate.
    """)
    return


@app.cell
def _(fit):
    report = fit.infer(covariance=HC("HC0")).report(reference="normal")
    mo.Html(report.to_html())
    return


@app.cell(hide_code=True)
def _(fit):
    mo.md(f"The estimated treatment slope is {float(fit.parameters[1]):.2f}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Choose a treatment value to inspect the structural mean, then compare the selected prediction with the fitted relationship across the range.
    """)
    return


@app.cell
def _():
    treatment_value = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.0, label="Treatment")
    treatment_value
    return (treatment_value,)


@app.cell
def _(fit, treatment_value):
    mo.md(f"Structural mean: {float(fit.predict({"treatment": [treatment_value.value]})[0]):.2f}")
    return


@app.cell(hide_code=True)
def _(fit, treatment_value):
    _x = jnp.linspace(-2.0, 2.0, 60)
    _y = np.asarray(fit.predict({"treatment": _x}))
    _selected = float(fit.predict({"treatment": [treatment_value.value]})[0])
    _fig, _ax = plt.subplots()
    _ax.plot(_x, _y)
    _ax.scatter([treatment_value.value], [_selected])
    _ax.set(xlabel="Treatment", ylabel="Structural mean")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The fitted result keeps the structural design for prediction and pairs the estimated treatment coefficient with the requested covariance for inference.
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
    mo.callout(mo.md('See the [assumptions and limits](../reference/moment-estimators.md#assumptions-and-limits) and [evidence](../reference/moment-estimators.md#evidence).'), kind="info")
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
    z, shock, noise = jax.random.normal(jax.random.key(412), (3, 128))
    treatment = 0.8 * z + 0.5 * shock
    return Observations({"y": 0.2 + 0.7 * treatment + 0.3 * shock + noise, "treatment": treatment, "assignment": z})


@app.cell(hide_code=True)
def _(fit):
    def test_fit_is_valid():
        assert bool(fit.valid)

    return


if __name__ == "__main__":
    app.run()
