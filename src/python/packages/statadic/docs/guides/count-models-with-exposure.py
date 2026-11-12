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
"""Fit counts with exposure and positive outcomes."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.data import Observations
    from statadic.api.design import Intercept, Linear
    from statadic.api.inference import ModelBased
    from statadic.api.regression import Gamma, Observed, Poisson, Regression
    from statadic.estimation.likelihood import FixedDispersion
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
    # Fit counts with exposure

    Fit a Poisson rate model with exposure and compare it with a Gamma model for positive outcomes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    Counts collected over different observation times need their exposure recorded alongside the outcome. A Poisson model uses that exposure to distinguish the rate from the expected count, while a Gamma model can describe a positive continuous response.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit the count model

    The generated counts use a known rate and increasing exposure, reflecting the longer observation periods available at higher load. The model treats exposure as an offset and estimates the effect of load on the log rate.
    """)
    return


@app.cell
def _():
    data, positive = make_data(23)
    fit = Regression(Observed("count"), Intercept() + Linear("load"), exposure="exposure", law=Poisson()).estimate(data).require_valid()
    rate = fit.predict(data, kind="rate")
    mo.plain_text(fit.report().to_text())
    return data, fit, positive, rate


@app.cell(hide_code=True)
def _(data, rate):
    _fig, _ax = plt.subplots()
    _ax.scatter(data.numeric("load"), data.numeric("count") / data.numeric("exposure"), s=14, alpha=0.5, label="Observed count / exposure")
    _ax.plot(data.numeric("load"), rate, label="Estimated rate")
    _ax.set(xlabel="Load", ylabel="Rate", title="Counts adjusted for exposure")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Fit a positive response

    The second outcome is continuous and strictly positive, so the Gamma law describes it with an explicitly fixed dispersion. This is a separate response from the counts in the first model.
    """)
    return


@app.cell
def _(data, positive):
    gamma_data = Observations({"load": data.numeric("load"), "response": positive})
    gamma_fit = Regression(Observed("response"), Intercept() + Linear("load"), law=Gamma(dispersion=FixedDispersion(0.6))).estimate(gamma_data).require_valid()
    gamma_intervals = gamma_fit.infer(covariance=ModelBased()).intervals(reference="normal")
    mo.plain_text(gamma_fit.report().to_text())
    return gamma_fit, gamma_intervals


@app.cell(hide_code=True)
def _(fit, gamma_fit):
    mo.md(f"The fitted load coefficients are **{float(fit.parameters[1]):.3f}** for the count model and **{float(gamma_fit.parameters[1]):.3f}** for the positive response. They describe separate generated outcomes.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how changing the exposure affects the expected count at the same
    load, then move the control. The rate per unit of exposure stays on its
    own scale.
    """)
    return


@app.cell
def _():
    exposure = mo.ui.slider(0.5, 3.0, step=0.25, value=1.0, label="Exposure")
    exposure
    return (exposure,)


@app.cell
def _(exposure, fit):
    _new = {"load": [1.0], "exposure": [exposure.value]}
    _rate = fit.predict(_new, kind="rate")
    _mean = fit.predict(_new, kind="mean")
    mo.md(f"Rate: **{float(_rate[0]):.3f}** · Expected count: **{float(_mean[0]):.3f}**")
    return


@app.cell(hide_code=True)
def _(exposure, fit):
    _new = {"load": [1.0], "exposure": [exposure.value]}
    _rate = float(fit.predict(_new, kind="rate")[0])
    _mean = float(fit.predict(_new, kind="mean")[0])
    _fig, _ax = plt.subplots()
    _ax.bar(["Rate", "Expected count"], [_rate, _mean])
    _ax.set_ylabel("Rate or expected count")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    Use the rate for a per-unit-exposure prediction and the mean when the expected number of events is needed. Both predictions come from the fitted model, but their interpretation depends on the exposure supplied for the new observations and on the units in which it is measured.
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
    mo.callout(mo.md('See the [assumptions and limits](../reference/conditional-laws.md#assumptions-and-limits) and [evidence](../reference/conditional-laws.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Reporting](../reference/reporting.md).
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
    """Generate reproducible count and positive continuous responses."""
    load = jnp.linspace(0.0, 2.0, 80)
    exposure = 1.0 + load / 2.0
    mean = exposure * jnp.exp(0.5 + 0.4 * load)
    key_count, key_gamma = jax.random.split(jax.random.key(seed))
    count = jax.random.poisson(key_count, mean)
    positive = 0.6 * jnp.exp(0.2 + 0.4 * load) * jax.random.gamma(key_gamma, 1 / 0.6, shape=load.shape)
    return Observations({"load": load, "exposure": exposure, "count": count}), positive


@app.cell(hide_code=True)
def _(gamma_intervals, rate):
    def test_result():
        assert rate.shape == (80,)
        assert bool(jnp.all(jnp.isfinite(rate)))
        assert bool(gamma_intervals.valid)

    return


if __name__ == "__main__":
    app.run()
