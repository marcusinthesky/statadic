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
    # Predict and retain the analysis

    A report carries computed estimates and their selected uncertainty into a table, while a saved predictor applies the fitted design to new inputs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A researcher studying how an outcome changes with one measured input can use these simulated observations to compare a fitted result with the line that generated them.

    The prediction uses the fitted conditional mean at each new input, and the report records the covariance and reference selected for inference. A saved snapshot can reproduce the presentation, while a saved predictor carries the design needed to evaluate new rows.
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
    ## Step 1: Predict new outcomes

    Prediction applies the fitted mean to new inputs using the design built from the training observations, keeping the original parameter order.
    """)
    return


@app.cell
def _(data):
    fit = fit_data(data)
    new_x = jnp.array([-1.0, 0.0, 1.0])
    predictions = fit.predict({"x": new_x})
    _fig, _ax = plt.subplots()
    _ax.scatter(data["x"], data["y"], s=14, label="Observations")
    _ax.scatter(new_x, predictions, marker="x", s=60, label="Predicted mean")
    _ax.set_xlabel("Input, x")
    _ax.set_ylabel("Outcome, y")
    _ax.legend()
    _ax
    return fit, predictions


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Render the selected report

    The HC3 report renders as HTML or LaTeX from the same calculated values, so the table stays connected to the selected covariance and reference. Use `report.save_latex(path)` for a table, `fit.save_predictor(path)` for prediction and `save_analysis(path, report)` for a report snapshot.
    """)
    return


@app.cell
def _(fit):
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    mo.vstack([mo.Html(report.to_html()), mo.plain_text(report.to_latex())])
    return


@app.cell(hide_code=True)
def _(predictions):
    mo.md(f"The predicted mean at zero is {float(predictions[1]):.2f}. This is a mean prediction rather than an interval for a future observation.")
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

    The fitted result applies its training design to new rows and connects the displayed intervals to the selected covariance and reference distribution.
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
    mo.callout(mo.md('See the [assumptions and limits](../../reference/reporting.md#assumptions-and-limits) and [evidence](../../reference/reporting.md#evidence).'), kind="info")
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
