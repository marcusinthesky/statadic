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
"""Include a checked mathematical result in a manuscript."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import numpy as np
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.design import Intercept, Linear
    from statadic.api.inference import HAC, HC, Clustered, ModelBased
    from statadic.api.regression import Regression
    from statadic.api.data import Observations
    from statadic.api.ordination import LinearGram, Ordination, PrincipalCoordinates
    from statadic.estimation.iv import TwoStageLeastSquares

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
            "axes.titlecolor": INK,
            "text.color": INK,
            "legend.labelcolor": INK,
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
    # Cite a checked result

    Carry the identity and assumptions of a mathematical result into a manuscript.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A checked statement is useful when the mathematical result in a paper matches the hypotheses established for the analysis. Record the claim identifier and source revision alongside those assumptions, so that a reader can trace the statement to its evidence.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Select a claim

    Select the MMD claim as an example, using its stable identifier to connect the statement with the registry.
    """)
    return


@app.cell
def _():
    claim = "STATADIC-MMD-001"
    mo.md(f"Selected claim: **{claim}**")
    return (claim,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Include the statement

    Download the macro registry from the mathematical statements page and include it in a Unicode-capable TeX document, where the claim identifier selects the checked statement.
    """)
    return


@app.cell
def _(claim):
    latex = "\\input{generated/statadic-mathematics.tex}\n\\statadicstatement{" + claim + "}\n\\statadicassumptions{" + claim + "}"
    mo.plain_text(latex)
    return (latex,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Choose which part of the claim to include in the example, then inspect the TeX command generated for that fragment.
    """)
    return


@app.cell
def _():
    part = mo.ui.dropdown(["statement", "assumptions", "diagnostics"], value="statement", label="Claim part")
    part
    return (part,)


@app.cell
def _(claim, part):
    mo.plain_text("\\statadic" + part.value + "{" + claim + "}")
    return


@app.cell(hide_code=True)
def _(part):
    _fig, _ax = plt.subplots()
    _labels = ["statement", "assumptions", "diagnostics"]
    _ax.bar(_labels, [int(_label == part.value) for _label in _labels])
    _ax.set(ylabel="Selected claim fragment", ylim=(0, 1.2), yticks=[0, 1])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The manuscript can present the statement and its assumptions with the same identifier used in the registry, giving readers a route back to its source.
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
    mo.callout(mo.md('Read the selected claim before applying its mathematical result. See the [assumptions and limits](../reference/energy.md#assumptions-and-limits) and [evidence](../reference/energy.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Formal evidence](../formal/index.md)
    - [Evidence and limitations](../validation/index.md)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.cell(hide_code=True)
def _(claim, latex):
    def test_claim_is_in_export():
        assert claim in latex

    return


if __name__ == "__main__":
    app.run()
