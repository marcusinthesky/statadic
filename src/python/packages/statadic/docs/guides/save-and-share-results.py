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
"""Save and share a fitted analysis."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
    from statadic.api.design import Intercept, Linear
    from statadic.api.inference import HC
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
    # Save and share results

    Render a report and reload its analysis snapshot while keeping the covariance choice visible.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A report carries estimates together with their inference choices, which makes the numbers easier to interpret when they leave your notebook, while a temporary directory lets this example run without leaving files behind.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit and report

    Choose the covariance estimator and reference distribution before exporting the report, since those choices determine the uncertainty columns that appear beside the estimates.
    """)
    return


@app.cell
def _():
    data = make_data(17)
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    mo.Html(report.to_html())
    return fit, report


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Export and reload

    Save a table in LaTeX and named values for a manuscript, a snapshot of the analysis for later rendering, and a predictor for new inputs. In your own project, replace the temporary path with a persistent results directory.
    """)
    return


@app.cell
def _(fit, report):
    with TemporaryDirectory() as _directory:
        _output = Path(_directory)
        report.save_latex(_output / "table.tex")
        report.save_values(_output / "values.tex", prefix="ols")
        fit.save_predictor(_output / "predictor")
        prediction = load_predictor(_output / "predictor").predict({"x": [1.0, 2.0]})
        save_analysis(_output / "analysis", report)
        restored = load_analysis(_output / "analysis").report
        table = (_output / "table.tex").read_text()
    mo.plain_text(f"Table characters: {len(table)}")
    return prediction, restored, table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Change the number of decimal places in the table and compare the displayed
    values, while the report preserves the same computed values underneath.
    """)
    return


@app.cell
def _():
    decimals = mo.ui.slider(1, 5, step=1, value=3, label="Decimal places")
    decimals
    return (decimals,)


@app.cell
def _(decimals, report):
    mo.plain_text(report.to_latex(decimals=decimals.value))
    return


@app.cell(hide_code=True)
def _(decimals, report):
    _estimate = float(report.parameters[1].estimate)
    _rounded = round(_estimate, decimals.value)
    _fig, _ax = plt.subplots()
    _ax.bar(["Stored", "Displayed"], [_estimate, _rounded])
    _ax.set_ylabel("Slope estimate")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The restored report can render the same computed values together with their interpretation, even after the original Python session has ended. A predictor restores prediction separately from inference, while the snapshot does not refit the model. Retain the original data and analysis code to repeat the computation.

    Include the table and named values in a manuscript using `booktabs` and `siunitx`. For a wide table, adjust the font size or column spacing in the manuscript that includes it. The [reporting reference](../reference/reporting.md) covers test tables and formatting options.
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
    mo.callout(mo.md('Inspect the loading and trust boundaries before sharing artifacts. See the [assumptions and limits](../reference/artifacts.md#assumptions-and-limits) and [evidence](../reference/artifacts.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Regression reference](../reference/regression.md).
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
    """Generate reproducible observations for the report."""
    x = jnp.linspace(0.0, 4.0, 40)
    y = 1.0 + 0.8 * x + 0.3 * jax.random.normal(jax.random.key(seed), x.shape)
    return {"x": x, "y": y}


@app.cell(hide_code=True)
def _(fit, prediction, report, restored, table):
    def test_result():
        assert restored.to_latex() == report.to_latex()
        assert len(table) > 0
        assert bool(jnp.allclose(prediction, fit.predict({"x": [1.0, 2.0]})))

    return


if __name__ == "__main__":
    app.run()
