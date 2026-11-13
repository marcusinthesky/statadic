# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.25.2",
#     "matplotlib>=3.8",
#     "wigglystuff>=0.5.40,<0.6",
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
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from wigglystuff import TangleSlider
    from statadic.api.design import Intercept, Linear
    from statadic.api.diagnostics import BreuschPagan
    from statadic.api.inference import HC
    from statadic.api.regression import Regression
    from statadic.api.artifacts import load_analysis, load_predictor, save_analysis

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
    # Your first regression

    Fit a straight line, look at what the residuals say about it, and report the
    coefficients with uncertainty you have chosen deliberately.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    Suppose a researcher measures how an outcome changes with a single input,
    such as reaction time against hours awake. Ordinary least squares draws
    the line that fits those measurements most closely, but the line on its
    own cannot say how much to trust its slope. That judgement depends on
    the residuals and on how the observations were collected.

    STATADIC keeps those decisions separate and visible. In this tutorial you
    will fit the line, inspect its residuals, request standard errors that
    remain reliable when the noise is uneven, and finish with a table you could
    place in a manuscript.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    You need only a little Python and the idea of a linear regression. Nothing
    here assumes earlier experience with JAX, which STATADIC uses for its
    numerical work behind the scenes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    The observations are simulated, so that we know the truth and can judge
    the results against it. Each outcome follows the line $y = 1 + 0.8x$, plus
    noise whose spread grows with $x$. That growing spread is common in real
    measurements, and it matters later when we choose how to measure
    uncertainty.
    """)
    return


@app.cell
def _():
    data = make_data(seed=17, n=60, spread=0.15)
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
    _ax.set_title("Sixty observations whose noise grows with x", loc="left")
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit the line

    A regression in STATADIC names its outcome and its terms explicitly, so the
    intercept is something you ask for rather than something added silently.
    Calling `estimate` fits the model to the data, and `require_valid` stops the
    analysis at once if the numerical fit could not be computed.
    """)
    return


@app.cell
def _(data):
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
    mo.ui.table([
        {"Parameter": row.name, "Estimate": row.estimate}
        for row in fit.report().parameters
    ])
    return (fit,)


@app.cell(hide_code=True)
def _(fit):
    _intercept, _slope = (float(value) for value in fit.parameters)
    mo.md(f"""
    The fitted line is $\\hat y = {_intercept:.2f} + {_slope:.2f}x$, close to the
    line that generated the data. So far, though, we have a point estimate only.
    The report deliberately contains no standard errors, because those depend
    on a choice we have not yet made.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Look at the residuals

    Residuals are what the line fails to explain. Plotting them against the
    input is the quickest way to see whether the spread is constant, which is
    the assumption behind the textbook standard errors.
    """)
    return


@app.cell
def _(data, fit):
    residuals = jnp.asarray(data["y"]) - fit.predict({"x": data["x"]})
    _fig, _ax = plt.subplots()
    _ax.axhline(0.0, color=INK, linewidth=0.8)
    _ax.scatter(data["x"], residuals, s=14)
    _ax.set_xlabel("Input, x")
    _ax.set_ylabel("Residual")
    _ax.set_title("The residuals fan out as x grows", loc="left")
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The fan shape suggests that the noise grows with the input. A formal test
    can put a number on that impression. The Breusch–Pagan test asks whether
    the squared residuals move with the regressors, and a small p-value is
    evidence that they do.
    """)
    return


@app.cell
def _(fit):
    diagnostics = fit.diagnose(BreuschPagan())
    breusch_pagan = next(iter(diagnostics))
    mo.ui.table([{
        "Test": "Breusch–Pagan",
        "Statistic": breusch_pagan.statistic,
        "p-value": breusch_pagan.pvalue,
        "Available": breusch_pagan.available,
    }])
    return (breusch_pagan,)


@app.cell(hide_code=True)
def _(breusch_pagan):
    mo.md(f"""
    The test returns a p-value of {float(breusch_pagan.pvalue):.3f}, which agrees
    with the picture. Under uneven noise the classical standard errors can be
    too small or too large, so the next step uses an estimator designed for
    this situation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Choose how to measure uncertainty

    Uncertainty is a separate request in STATADIC. You choose a covariance
    estimator and a reference distribution, and the fitted line stays exactly
    as it was. Here we use HC3, a heteroskedasticity-consistent estimator that
    behaves well in moderate samples, with intervals from the normal
    distribution.
    """)
    return


@app.cell
def _(fit):
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    mo.ui.table([
        {"Parameter": row.name, "Estimate": row.estimate, "HC3 SE": row.std_error,
         "Lower": row.ci_lower, "Upper": row.ci_upper}
        for row in report.parameters
    ])
    return (report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4: Save the result for a manuscript

    The same report renders as LaTeX, ready to `\input` into a paper. The table
    records the covariance estimator and reference distribution alongside the
    numbers, so a reader of the manuscript can see how they were produced.
    Save an analysis snapshot to reproduce that report and a predictor to
    apply the fitted design to new observations. This example uses temporary
    files that are removed when the cell finishes.
    """)
    return


@app.cell
def _(fit, report):
    with TemporaryDirectory() as _directory:
        _output = Path(_directory)
        report.save_latex(_output / "regression.tex")
        save_analysis(_output / "analysis", report)
        fit.save_predictor(_output / "predictor")
        restored_report = load_analysis(_output / "analysis").report
        prediction = load_predictor(_output / "predictor").predict({"x": [5.0]})
    mo.vstack([mo.plain_text(report.to_latex()), mo.plain_text(f"Predicted mean at x=5: {float(prediction[0]):.2f}")])
    return prediction, restored_report


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Before moving the slider, predict what will happen to the HC3 standard
    error of the slope as the noise spreads out more quickly. Then try it, and
    watch whether the Breusch–Pagan p-value moves in the direction you
    expected.
    """)
    return


@app.cell
def _():
    spread = mo.ui.anywidget(TangleSlider(amount=0.15, min_value=0.0, max_value=0.4, step=0.05, digits=2, prefix="Noise growth: "))
    spread
    return (spread,)


@app.cell
def _(spread):
    _data = make_data(seed=17, n=60, spread=spread.value["amount"])
    _fit = Regression.ols("y", Intercept() + Linear("x")).estimate(_data).require_valid()
    _test = next(iter(_fit.diagnose(BreuschPagan())))
    _report = _fit.infer(covariance=HC("HC3")).report(reference="normal")
    mo.vstack(
        [
            mo.md(f"Breusch–Pagan p-value: **{float(_test.pvalue):.3f}**"),
            mo.Html(_report.to_html()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(spread):
    explore_figure(spread)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - A model names its outcome and terms, and `estimate` returns a fitted
      result that you can check with `require_valid`.
    - Residual plots and diagnostics such as Breusch–Pagan test the assumptions
      behind the standard errors.
    - Uncertainty is a separate request, so you can change the covariance
      estimator without refitting the line.
    - Reports render as text, HTML and LaTeX from the same numbers.
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
    mo.callout(mo.md('HC3 intervals assume independent observations and a large sample. See the [assumptions and limits](../reference/regression.md#assumptions-and-limits) and [evidence](../reference/regression.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - The [complete OLS analysis](../tutorials/ols/index.md) adds influence,
      hypothesis tests and prediction.
    - [Change the covariance estimator](../guides/change-covariance.py).
    - Why [models, inference and reports stay separate](../concepts/index.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def explore_figure(spread):
    _data = make_data(seed=17, n=60, spread=spread.value["amount"])
    _fig, _ax = plt.subplots()
    _ax.scatter(_data["x"], _data["y"], s=14)
    _ax.set(xlabel="Input x", ylabel="Outcome y")
    return _fig


@app.function(hide_code=True)
def make_data(seed: int, n: int, spread: float) -> dict[str, list[float]]:
    """Draw a straight line whose noise grows linearly with x.

    Args:
        seed: Seed for the JAX random key.
        n: Number of observations.
        spread: How quickly the noise standard deviation grows with x.

    Returns:
        Columns ``x`` and ``y`` as Python lists.

    Examples:
        >>> data = make_data(seed=0, n=5, spread=0.1)
        >>> sorted(data), len(data["x"])
        (['x', 'y'], 5)
    """
    x = jnp.linspace(0.0, 10.0, n)
    noise = (0.3 + spread * x) * jax.random.normal(jax.random.key(seed), x.shape)
    return {"x": x.tolist(), "y": (1.0 + 0.8 * x + noise).tolist()}


@app.cell(hide_code=True)
def _(fit):
    def test_slope_is_close_to_the_generating_line():
        assert abs(float(fit.parameters[1]) - 0.8) < 0.15

    return


@app.cell(hide_code=True)
def _(breusch_pagan):
    def test_breusch_pagan_detects_the_growing_noise():
        assert float(breusch_pagan.pvalue) < 0.05

    return


@app.cell(hide_code=True)
def _(prediction, report, restored_report):
    def test_saved_results_preserve_report_and_prediction():
        assert restored_report.to_text() == report.to_text()
        assert prediction.shape == (1,)
        assert bool(jnp.all(jnp.isfinite(prediction)))

    return


if __name__ == "__main__":
    app.run()
