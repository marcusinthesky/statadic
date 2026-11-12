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
"""Compare least-squares estimators."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.data import RelativePrecision
    from statadic.api.design import Intercept, Linear
    from statadic.api.regression import Regression
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
    # Compare estimators

    Fit ordinary and weighted least squares to the same observations and compare their coefficients.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    When observations have different known precisions, weighted least squares gives more influence to the more precise measurements. Compare it with ordinary least squares on the same generated data to see the effect of that choice.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit both models

    The weights below are relative precisions derived from the noise scale, so their meaning is separate from frequency or survey weights. With more precise observations given greater weight, a small collection of noisy rows has less influence on the fitted line.
    """)
    return


@app.cell
def _():
    data = make_data(17)
    terms = Intercept() + Linear("x")
    ols = Regression.ols("y", terms).estimate(data).require_valid()
    wls = Regression("y", terms, precision=RelativePrecision("precision")).estimate(data).require_valid()
    mo.plain_text(f"OLS: {ols.parameters}\nWLS: {wls.parameters}")
    return data, ols, wls


@app.cell(hide_code=True)
def _(data, ols, wls):
    _fig, _ax = plt.subplots()
    _ax.scatter(data["x"], data["y"], s=15, alpha=0.7)
    _ax.plot(data["x"], ols.predict({"x": data["x"]}), label="OLS")
    _ax.plot(data["x"], wls.predict({"x": data["x"]}), label="WLS")
    _ax.set(xlabel="x", ylabel="y", title="Same rows, different precision weights")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(ols, wls):
    mo.md(f"The estimated slopes are **{float(ols.parameters[1]):.3f}** for OLS and **{float(wls.parameters[1]):.3f}** for WLS. The difference reflects how each estimator weights the same rows.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how the fitted slope changes as the differences in row precision
    become less pronounced, then move the control to change the relative
    weights while keeping the observations and design fixed.
    """)
    return


@app.cell
def _():
    weight_power = mo.ui.slider(0.0, 1.0, step=0.25, value=1.0, label="Strength of precision weighting")
    weight_power
    return (weight_power,)


@app.cell
def _(data, weight_power):
    _weighted_data = dict(data, precision=data["precision"] ** weight_power.value)
    _fit = Regression("y", Intercept() + Linear("x"), precision=RelativePrecision("precision")).estimate(_weighted_data).require_valid()
    mo.md(f"The slope is **{float(_fit.parameters[1]):.3f}** at weight power **{weight_power.value:.2f}**.")
    return


@app.cell(hide_code=True)
def _(data, weight_power):
    _weighted = dict(data, precision=data["precision"] ** weight_power.value)
    _fit = Regression("y", Intercept() + Linear("x"), precision=RelativePrecision("precision")).estimate(_weighted).require_valid()
    _fig, _ax = plt.subplots()
    _ax.scatter(data["x"], data["y"], s=14, alpha=0.6)
    _ax.plot(data["x"], _fit.predict({"x": data["x"]}))
    _ax.set(xlabel="Input x", ylabel="Outcome y")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The fits use the same design but answer different weighted estimation problems, so the coefficients need not be identical. Request uncertainty separately for the fitted result you intend to report, using an estimator suited to the sampling design.
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
    mo.callout(mo.md('Precision weights have a distinct interpretation from frequency or survey weights. See the [assumptions and limits](../reference/regression.md#assumptions-and-limits) and [evidence](../reference/regression.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Change covariance](change-covariance.py).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def make_data(seed: int):
    """Generate a line with known, uneven observation variance."""
    x = jnp.linspace(0.0, 4.0, 40)
    sigma = 0.3 + 0.3 * x
    y = 1.0 + 0.8 * x + sigma * jax.random.normal(jax.random.key(seed), x.shape)
    return {"x": x, "y": y, "precision": 1 / sigma**2}


@app.cell(hide_code=True)
def _(ols, wls):
    def test_result():
        assert bool(ols.valid) and bool(wls.valid)
        assert bool(jnp.all(jnp.isfinite(wls.parameters)))

    return


if __name__ == "__main__":
    app.run()
