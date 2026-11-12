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
"""Change the covariance estimator for a fitted regression."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.design import Intercept, Linear
    from statadic.api.inference import HAC, HC, Clustered, ModelBased
    from statadic.api.regression import Regression

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
    # Change the covariance estimator

    Recompute the standard errors of a fitted regression under a different
    covariance estimator, without refitting the line.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    The right standard errors depend on how the observations were collected.
    Independent rows with uneven noise call for a heteroskedasticity-consistent
    estimator, repeated measurements of the same firm call for clustering, and
    observations ordered in time call for an estimator that allows for
    autocorrelation. In STATADIC each of these is a request made of the same
    fitted result, so the coefficients stay fixed while the uncertainty
    changes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    This guide assumes you have fitted a regression before, as in
    [your first regression](../start/first-regression.py).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit the model once

    The data contain an outcome, a predictor, a firm label and a time index.
    Columns that are not terms of the model, such as `firm` and `time`, stay
    with the fitted result, so that later covariance requests can use them.
    """)
    return


@app.cell
def _():
    data = make_panel(seed=3, firms=20, periods=12)
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Request each estimator from the same fit

    Pass a covariance request to `infer`, then choose the reference
    distribution when you build the report. `ModelBased` gives the classical
    standard errors, `HC` the heteroskedasticity-consistent family, `Clustered`
    groups rows by a label, and `HAC` allows for correlation between nearby
    observations in a stated order.
    """)
    return


@app.cell
def _(fit):
    requests = {
        "Classical": ModelBased(),
        "HC3": HC("HC3"),
        "Clustered by firm": Clustered("firm"),
        "HAC, bandwidth 3": HAC(bandwidth=3, order_by="time"),
    }
    reports = {
        name: fit.infer(covariance=request).report(reference="normal")
        for name, request in requests.items()
    }
    mo.ui.tabs({name: mo.Html(report.to_html()) for name, report in reports.items()})
    return (reports,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Compare the standard errors

    Placing the slope's standard errors side by side shows how much the choice
    matters. In these data the same firm appears in every period, so its
    shared shocks make the clustered standard error the largest.
    """)
    return


@app.cell
def _(reports):
    slope_errors = {name: slope_standard_error(report) for name, report in reports.items()}
    _fig, _ax = plt.subplots()
    _ax.barh(list(slope_errors), list(slope_errors.values()), color=BLUE)
    _ax.invert_yaxis()
    _ax.set_xlabel("Standard error of the slope")
    _ax
    return (slope_errors,)


@app.cell(hide_code=True)
def _(slope_errors):
    _ratio = slope_errors["Clustered by firm"] / slope_errors["Classical"]
    mo.md(f"""
    Clustering by firm makes the slope's standard error {_ratio:.1f} times the
    classical value, while the estimated slope itself is identical in every
    report.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    You now have four reports from one fitted model, each recording which
    covariance estimator and reference distribution produced its intervals.
    Choose the one that matches how your observations were collected, and
    report that choice alongside the numbers.
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
    mo.callout(mo.md('Each estimator relies on its own large-sample argument. See the [assumptions and limits](../reference/regression.md#assumptions-and-limits) and [evidence](../reference/regression.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Covariance and reference distributions](../concepts/index.md) explains
      why uncertainty is a separate request.
    - The [regression reference](../reference/regression.md) lists every
      covariance option.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def make_panel(seed: int, firms: int, periods: int) -> dict[str, list[float]]:
    """Simulate firms observed over time with a shared shock per firm.

    Args:
        seed: Seed for the JAX random key.
        firms: Number of firms.
        periods: Number of periods observed for each firm.

    Returns:
        Columns ``y``, ``x``, ``firm`` and ``time``, one row per firm and period.

    Examples:
        >>> panel = make_panel(seed=0, firms=2, periods=3)
        >>> sorted(panel), len(panel["y"])
        (['firm', 'time', 'x', 'y'], 6)
    """
    key_x, key_firm, key_noise = jax.random.split(jax.random.key(seed), 3)
    firm = jnp.repeat(jnp.arange(firms), periods)
    time = jnp.tile(jnp.arange(periods), firms)
    firm_x = jax.random.normal(key_x, (firms,))
    x = firm_x[firm] + 0.3 * jax.random.normal(key_noise, firm.shape)
    shock = jax.random.normal(key_firm, (firms,))[firm]
    y = 1.0 + 0.5 * x + shock + 0.5 * jax.random.normal(key_noise, firm.shape)
    return {
        "y": y.tolist(),
        "x": x.tolist(),
        "firm": firm.tolist(),
        "time": (firm * periods + time).astype(float).tolist(),
    }


@app.function(hide_code=True)
def slope_standard_error(report) -> float:
    """Return the standard error of the slope ``x`` from a report.

    Args:
        report: A regression report with a ``parameters`` table.

    Returns:
        The standard error of the coefficient named ``x``.
    """
    return next(float(row.std_error) for row in report.parameters if row.name == "x")


@app.cell(hide_code=True)
def _(slope_errors):
    def test_clustering_widens_the_slope_interval():
        assert slope_errors["Clustered by firm"] > slope_errors["Classical"]

    return


if __name__ == "__main__":
    app.run()
