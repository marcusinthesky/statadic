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
"""Self-contained tutorial using seeded simulated data."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from statadic.discrepancy.mmd.energy.geometry import resolve_energy_geometry
    from statadic.inference.resampling.permutation import energy_permutation_test_result

    DARK = mo.app_meta().theme == "dark"
    INK = "#f2f2f2" if DARK else "#000000"
    BLUE = "#8fb8ff" if DARK else "#1d3f8f"
    RULE = "#555555" if DARK else "#d6d6d6"
    plt.rcParams.update({
        "figure.figsize": (6.4, 3.6), "figure.facecolor": "none",
        "axes.facecolor": "none", "savefig.format": "svg", "savefig.transparent": True,
        "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "text.color": INK, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "axes.prop_cycle": plt.cycler(color=[BLUE, "#8a8a8a", INK]),
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": RULE, "grid.linewidth": 0.6,
        "xtick.color": INK, "ytick.color": INK,
    })


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Test a difference with permutations

    Compare two samples with an energy statistic and shuffled labels.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    Imagine two groups of measurements drawn under different conditions, and ask whether their observed separation is unusual under a shared distribution. We will compare that separation with shuffled group assignments, then interpret the resulting p-value in the context of the experiment.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    Read the [energy reference](../reference/energy.md) for the distance between distributions, including the ground distance that determines which differences the statistic can detect.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    Two independent normal samples contain 32 observations each, with a location shift of 0.75 in the second group. A fixed seed reproduces the observations, so the displayed histogram and the later test use the same data.
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32, 1))
    y = rng.normal(loc=0.75, size=(32, 1))
    _fig, _ax = plt.subplots()
    _ax.hist(x[:, 0], bins=10, alpha=0.5, label="First group")
    _ax.hist(y[:, 0], bins=10, alpha=0.5, label="Second group")
    _ax.set_xlabel("Measurement")
    _ax.set_ylabel("Observations")
    _ax.legend()
    _preview = mo.ui.table([{"first": float(x[_i, 0]), "second": float(y[_i, 0])} for _i in range(5)])
    mo.vstack([_preview, _fig])
    return x, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Compute the permutation distribution

    The energy statistic measures separation between the groups by comparing distances within and between their samples. Reassigning labels builds a reference distribution for exchangeable observations, with each shuffled result measured on the same scale as the original statistic.
    """)
    return


@app.cell
def _(x, y):
    result = energy_test(x, y, seed=43, permutations=99)
    _fig, _ax = plt.subplots()
    _ax.hist(np.asarray(result.null_distribution), bins=12)
    _ax.axvline(float(result.statistic), color=INK, label="Observed")
    _ax.set_xlabel("Scaled energy statistic")
    _ax.set_ylabel("Permutations")
    _ax.legend()
    _fig
    return (result,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Read the tail probability

    Count shuffled statistics at least as large as the observed statistic, including ties in the upper tail. The add-one correction includes the observed arrangement and determines the smallest p-value that this number of permutations can report.
    """)
    return


@app.cell
def _(result):
    exceedances = int(np.count_nonzero(np.asarray(result.null_distribution) >= float(result.statistic)))
    mo.md(f"The observed statistic is **{float(result.statistic):.3f}**, with "
          f"**{exceedances}** exceedances among 99 permutations and "
          f"an add-one p-value of **{float(result.pvalue):.3f}**.")
    return (exceedances,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Before moving the control, predict how the smallest reportable p-value changes with the number of permutations, and consider how much detail 19 draws allow.
    """)
    return


@app.cell
def _():
    count = mo.ui.slider(19, 199, step=20, value=99, label="Permutations")
    count
    return (count,)


@app.cell
def _(count, x, y):
    _result = energy_test(x, y, seed=43, permutations=count.value)
    _fig, _ax = plt.subplots()
    _ax.hist(np.asarray(_result.null_distribution), bins=12)
    _ax.axvline(float(_result.statistic), color=INK)
    _ax.set_xlabel("Scaled energy statistic")
    _ax.set_ylabel("Permutations")
    mo.vstack([mo.md(f"p = {float(_result.pvalue):.3f}. Minimum = {1 / (count.value + 1):.3f}"), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - A permutation distribution compares the observed split with the statistics obtained from shuffled group labels under the specified randomization scheme.
    - More permutations improve p-value resolution, although they cannot make an unsuitable shuffling design valid.
    - A p-value measures tail evidence under the null, rather than the probability that the null is true.
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
    mo.callout(mo.md('Exchangeability and the permutation schedule determine the interpretation. See the [assumptions and limits](../reference/resampling.md#assumptions-and-limits) and [evidence](../reference/resampling.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Study calibration](calibration.py) across repeated samples.
    - [Review energy distances](../reference/energy.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def energy_test(x, y, seed, permutations):
    """Compare independent samples with a seeded label permutation test."""
    return energy_permutation_test_result(
        jnp.asarray(x), jnp.asarray(y), resolve_energy_geometry("euclidean", 1.0),
        jax.random.PRNGKey(seed), num_permutations=permutations, alternative="greater",
    )


@app.cell(hide_code=True)
def _(exceedances, result):
    def test_add_one_pvalue():
        assert abs(float(result.pvalue) - (1 + exceedances) / 100) < 1e-6

    return


if __name__ == "__main__":
    app.run()
