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
"""Your first regression with STATADIC."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.design import Intercept, Linear
    from statadic.api.diagnostics import BreuschPagan, BreuschGodfrey, DurbinWatson, Influence, JarqueBera, RESET, White
    from statadic.api.inference import HAC, HC, ModelBased
    from statadic.api.data import Observations
    from statadic.api.hypotheses import Hypothesis, LinearRestriction
    from statadic.api.regression import Regression

    DARK = mo.app_meta().theme == "dark"
    INK = "#eeeeee" if DARK else "#000000"
    BLUE = "#90b4ff" if DARK else "#1d3f8f"
    RULE = "#555555" if DARK else "#d6d6d6"
    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 3.6),
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
            "axes.prop_cycle": plt.cycler(color=[BLUE, "#8a8a8a", INK]),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Estimate and inspect the line

    A fitted line summarizes the relationship between an input and an outcome.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    When a researcher measures an outcome against a single input, a fitted line summarizes how the outcome changes across that input. This simulated sample makes the generating line known, so that you can compare the estimated coefficients with their target and see the effect of sampling noise.

    The intercept shifts the line. The residual degrees of freedom count the observations left after estimating the independent columns of that design.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    The [first regression](../../start/first-regression.py) introduces the model and its fitted result.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    The 80 observations follow $y = 1 + 2x$ with seeded Gaussian noise, and the `time` column records their order for diagnostics and covariance.
    """)
    return


@app.cell
def _():
    data = make_data()
    return (data,)


@app.cell(hide_code=True)
def _(data):
    mo.ui.table([
        {"x": float(data["x"][i]), "y": float(data["y"][i])}
        for i in range(5)
    ])
    return


@app.cell(hide_code=True)
def _(data):
    _fig, _ax = plt.subplots()
    _ax.scatter(data["x"], data["y"], s=14)
    _ax.set_xlabel("Input, x")
    _ax.set_ylabel("Outcome, y")
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Estimate the coefficients

    Name the outcome and both terms in the design, then check that the numerical fit is valid before interpreting its coefficients.
    """)
    return


@app.cell
def _(data):
    fit = fit_data(data)
    mo.Html(fit.report().to_html())
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Compare the line with the observations

    Plot the fitted mean at each input alongside the observations, so that the line can be compared with the pattern in the data.
    """)
    return


@app.cell
def _(data, fit):
    _fig, _ax = plt.subplots()
    _ax.scatter(data["x"], data["y"], s=14, label="Observations")
    _ax.plot(data["x"], fit.predict({"x": data["x"]}), label="Fitted mean")
    _ax.set_xlabel("Input, x")
    _ax.set_ylabel("Outcome, y")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(fit):
    mo.md(f"The slope is {float(fit.parameters[1]):.2f}, compared with 2 in the generating line.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how increasing the noise will affect the fitted line, then move the slider to compare the line with the observations.
    """)
    return


@app.cell
def _():
    noise = mo.ui.slider(0.1, 2.0, step=0.1, value=0.6, label="Noise standard deviation")
    noise
    return (noise,)


@app.cell
def _(noise):
    _data = make_data(noise=noise.value)
    _fit = Regression.ols("y", Intercept() + Linear("x")).estimate(_data).require_valid()
    _fig, _ax = plt.subplots()
    _ax.scatter(_data["x"], _data["y"], s=14)
    _ax.plot(_data["x"], _fit.predict({"x": _data["x"]}))
    _ax.set_xlabel("Input, x")
    _ax.set_ylabel("Outcome, y")
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    The fitted result connects the observations to the estimated line, and its report records the quantities available from the fit. The next chapter examines the residuals to ask where that line leaves systematic patterns.
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
    mo.callout(mo.md('See the [assumptions and limits](../../reference/regression.md#assumptions-and-limits) and [evidence](../../reference/regression.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [OLS chapter](index.md)
    - [Linear diagnostics](../../reference/linear-diagnostics.md)
    - [Reports](../../reference/reporting.md)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def make_data(noise: float = 0.6) -> dict:
    """Generate a reproducible linear sample with a positional time key."""
    x = np.linspace(-2, 2, 80)
    y = 1 + 2 * x + noise * np.random.default_rng(17).normal(size=len(x))
    return {"x": x.tolist(), "y": y.tolist(), "time": list(range(len(x)))}


@app.function(hide_code=True)
def fit_data(data):
    """Estimate the line from the supplied observations."""
    return Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()


@app.cell(hide_code=True)
def _(fit):
    def test_slope_near_generating_value():
        assert abs(float(fit.parameters[1]) - 2.0) < 0.2

    return


if __name__ == "__main__":
    app.run()
