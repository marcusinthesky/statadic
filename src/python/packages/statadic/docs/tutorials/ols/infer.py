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
    # Choose covariance and reference separately

    A fitted slope becomes more useful for inference when it is paired with an uncertainty calculation and a stated reference distribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A researcher studying how an outcome changes with one measured input can use these simulated observations to compare a fitted result with the line that generated them.

    Model-based covariance uses an estimate of the common variance for errors in the observations, whereas HC3 adjusts each residual contribution for leverage. The HAC calculation also includes products across nearby ordered observations, which makes the order and bandwidth part of the request.
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
    ## Step 1: Compare standard errors

    The same fitted coefficients can be paired with model-based, HC3 or ordered HAC covariance estimates, each addressing a different source of sampling variation in the observations.
    """)
    return


@app.cell
def _(data):
    fit = fit_data(data)
    classical = fit.infer(covariance=ModelBased())
    robust = fit.infer(covariance=HC("HC3"))
    hac = fit.infer(covariance=HAC(3, order_by="time"))
    mo.vstack([mo.md("**Model-based, t reference**"), mo.Html(classical.report(reference="t", degrees_of_freedom=fit.df_resid).to_html()), mo.md("**HC3, normal reference**"), mo.Html(robust.report(reference="normal").to_html()), mo.md("**HAC, normal reference**"), mo.Html(hac.report(reference="normal").to_html())])
    return fit, robust


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Test a named coefficient

    A restriction names the coefficient under test, and the reference law is chosen separately from the covariance estimator used to measure its uncertainty. The HAC bandwidth counts adjacent observations in ascending `time`, while the normal references for HC3 and HAC use approximations based on large samples.
    """)
    return


@app.cell
def _(robust):
    hypothesis = Hypothesis(LinearRestriction.zero("x"))
    outcome = robust.test(hypothesis, reference="chi2")
    mo.Html(outcome.report().to_html())
    return (outcome,)


@app.cell(hide_code=True)
def _(fit, outcome):
    mo.md(f"The slope estimate is {float(fit.parameters[1]):.2f}, and the HC3 test gives a p-value of {float(outcome.pvalue):.3g}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how increasing the noise will affect the estimated slope and its uncertainty. Move the slider and compare the fitted line with the observations.
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
    _report = _fit.infer(covariance=HC("HC3")).report(reference="normal")
    _slope = _report.parameters[1]
    mo.vstack([mo.md(f"Slope: {float(_slope.estimate):.3f}, HC3 standard error: {float(_slope.std_error):.3f}"), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    The fitted result connects each covariance request with the same estimated coefficients, while the reference distribution supplies a separate basis for intervals and tests.
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
