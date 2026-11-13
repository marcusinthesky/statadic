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
    # Ask questions of the residuals

    Residuals show where the fitted line misses the observations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A researcher studying how an outcome changes with one measured input can use these simulated observations to compare a fitted result with the line that generated them.

    The tests address different alternatives, so interpret their results alongside the plot of residuals and within the context of the study. A large p-value is compatible with many explanations, including a sample too small to reveal a departure.
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
    ## Step 1: Plot the residuals

    A plot of residuals against the input can reveal curvature or changing spread that a single fit statistic conceals, giving the numerical diagnostics a useful visual context for the observations in this sample.
    """)
    return


@app.cell
def _(data):
    fit = fit_data(data)
    residuals = jnp.asarray(data["y"]) - fit.predict({"x": data["x"]})
    _fig, _ax = plt.subplots()
    _ax.axhline(0, color=INK, linewidth=0.8)
    _ax.scatter(data["x"], residuals, s=14)
    _ax.set_xlabel("Input, x")
    _ax.set_ylabel("Residual")
    _ax
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Request diagnostics

    Each diagnostic asks a specific question about the residuals, while influence measures how sensitive the fitted coefficients are to individual observations. Breusch–Godfrey uses two ordered lags, and Durbin–Watson summarizes adjacent differences without assigning a p-value. Large influence invites investigation of the observation and its role in the analysis.
    """)
    return


@app.cell
def _(fit):
    diagnostics = fit.diagnose(JarqueBera(), BreuschPagan(), White(), BreuschGodfrey(lags=2, order_by="time"), DurbinWatson(order_by="time"), RESET(), Influence())
    mo.Html(diagnostics.report().to_html())
    return


@app.cell(hide_code=True)
def _(fit):
    mo.md(f"The residual sum of squares is {float(fit.core.sse):.2f}, providing a measure of overall error to consider alongside the plot and individual diagnostics.")
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

    A fitted result connects each diagnostic with the data and model that produced its residuals. Use those results alongside the plot and your understanding of how the observations were collected.
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
    mo.callout(mo.md('See the [assumptions and limits](../../reference/linear-diagnostics.md#assumptions-and-limits) and [evidence](../../reference/linear-diagnostics.md#evidence).'), kind="info")
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
