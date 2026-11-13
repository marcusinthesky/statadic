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
"""Scaffold for STATADIC tutorial and how-to notebooks.

Copy this file, keep every section heading in order, and replace the bracketed
guidance. Each heading lives in its own pure ``mo.md`` cell so that it exports
as Markdown and passes the page schema.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.design import Intercept, Linear
    from statadic.api.inference import HC
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
    # [Title: what the reader will do]

    [In one plain sentence, state the outcome of this notebook and explain what the reader will produce by following its steps and running its examples.]
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    [Open with the question and why it matters to a researcher, using a concrete
    example before the abstraction and stating what the reader will have
    produced by the end of the notebook. Explain how the example connects
    the observed data to the result the reader will inspect.]
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    [Name the knowledge this notebook assumes and link to the pages that teach
    it. Installation belongs to the page chrome, so leave it out.]
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    [Describe where the observations come from and what each column means.]
    """)
    return


@app.cell
def _():
    data = make_data(seed=17, n=40, noise=0.4)
    return (data,)


@app.cell(hide_code=True)
def _(data):
    mo.ui.table([{"x": float(data["x"][i]), "y": float(data["y"][i])} for i in range(5)])
    return


@app.cell
def _(data):
    _fig, _ax = plt.subplots()
    _ax.scatter(data["x"], data["y"], s=14)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit the line

    [One or two sentences on why this step comes first.]
    """)
    return


@app.cell
def _(data):
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
    mo.plain_text(fit.report().to_text())
    return (fit,)


@app.cell(hide_code=True)
def _(fit):
    _slope = float(fit.parameters[1])
    mo.md(f"""
    [Interpret the output with computed values, so the prose cannot drift from
    the result: the fitted slope is {_slope:.2f}.]
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Request uncertainty

    [Explain the choice this step makes and what it adds to the fit.]
    """)
    return


@app.cell
def _(fit):
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    mo.plain_text(report.to_text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    [Pose a question for the reader to predict before moving a control, then
    let the controls rerun the steps above. The static page shows the default
    values, so they must match the narrative.]
    """)
    return


@app.cell
def _():
    noise = mo.ui.slider(0.1, 2.0, step=0.1, value=0.4, label="Noise")
    noise
    return (noise,)


@app.cell
def _(noise):
    _data = make_data(seed=17, n=40, noise=noise.value)
    _fit = Regression.ols("y", Intercept() + Linear("x")).estimate(_data)
    mo.plain_text(_fit.infer(covariance=HC("HC3")).report(reference="normal").to_text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - [Three to five short bullets, each one idea.]
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
    mo.callout(mo.md('See the [assumptions and limits](../reference/regression.md#assumptions-and-limits) and [evidence](../reference/regression.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Two to four links to the natural next pages.]
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def make_data(seed: int, n: int, noise: float) -> dict[str, list[float]]:
    """Draw a straight line with Gaussian noise.

    Args:
        seed: Seed for the JAX random key.
        n: Number of observations.
        noise: Standard deviation of the Gaussian noise.

    Returns:
        Columns ``x`` and ``y`` as Python lists.

    Examples:
        >>> sorted(make_data(seed=0, n=5, noise=0.1))
        ['x', 'y']
    """
    x = jnp.linspace(0.0, 5.0, n)
    y = 1.0 + 0.8 * x + noise * jax.random.normal(jax.random.key(seed), x.shape)
    return {"x": x.tolist(), "y": y.tolist()}


@app.cell(hide_code=True)
def _(fit):
    def test_slope_is_close_to_the_generator():
        assert abs(float(fit.parameters[1]) - 0.8) < 0.3

    return


if __name__ == "__main__":
    app.run()
