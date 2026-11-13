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
    # Study calibration and estimator bias

    Repeat simulated experiments to examine rejection rates and MMD estimator bias.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A single p-value describes one dataset, while repeating the entire experiment reveals how often a test rejects under a chosen null and specific alternatives. Each trial uses independent samples, so the resulting frequencies describe this particular simulation design.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    Work through [permutation tests](permutation-tests.py) to see how the reference distribution is constructed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    Each simulated experiment draws independent normal samples with 24 observations per group, using shifts of 0, 0.75 and 1.5. Separate seeded streams choose the data and permutations, so an experiment can be reproduced without reusing its random draws.
    """)
    return


@app.cell
def _():
    shifts = (0.0, 0.75, 1.5)
    _rng = np.random.default_rng(123)
    _example = [_rng.normal(loc=s, size=24) for s in shifts]
    _fig, _ax = plt.subplots()
    _ax.boxplot(_example, tick_labels=[str(s) for s in shifts])
    _ax.set_xlabel("Location shift")
    _ax.set_ylabel("Measurement")
    _preview = mo.ui.table([{str(_s): float(_example[_j][_i]) for _j, _s in enumerate(shifts)} for _i in range(5)])
    mo.vstack([_preview, _fig])
    return (shifts,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Repeat the test

    Run 20 independent trials per shift with 49 permutations each, using the first scenario to estimate rejection frequency under the null and the others to describe power for these particular alternatives. The displayed intervals reflect Monte Carlo uncertainty from the limited number of trials, so repeat the experiment with a larger budget before drawing conclusions about small differences.
    """)
    return


@app.cell
def _(shifts):
    study = calibration(shifts, trials=20, permutations=49, seed=123)
    _fig, _ax = plt.subplots()
    _ax.errorbar(shifts, study[:, 0], yerr=[study[:, 0] - study[:, 1], study[:, 2] - study[:, 0]], fmt="o")
    _ax.axhline(0.05, color=INK, linestyle="--", label="Nominal level")
    _ax.set_xlabel("Location shift")
    _ax.set_ylabel("Rejection frequency")
    _ax.set_ylim(0, 1)
    _ax.legend()
    _fig
    return (study,)


@app.cell(hide_code=True)
def _(study):
    mo.md(f"The null rejected in **{study[0, 0]:.0%}** of trials. "
          f"Its approximate 95% Wilson interval is **[{study[0, 1]:.2f}, {study[0, 2]:.2f}]**. "
          "This interval describes uncertainty in the rejection frequency.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Compare U and V estimates

    For Gaussian-kernel MMD squared, the U estimate omits within-sample diagonals while the V estimate includes them, changing their behavior in finite samples. Compare their average bias with the known population value for normal samples.
    """)
    return


@app.cell
def _():
    bias = bias_study(trials=20, seed=123)
    _fig, _ax = plt.subplots()
    _pos = np.arange(len(bias))
    _ax.bar(_pos - 0.18, bias[:, 0], 0.36, yerr=bias[:, 2], label="U")
    _ax.bar(_pos + 0.18, bias[:, 1], 0.36, yerr=bias[:, 3], label="V")
    _ax.axhline(0, color=INK, linewidth=0.8)
    _ax.set_xticks(_pos, ["null equal", "null unequal", "shift equal", "shift unequal"], rotation=20)
    _ax.set_ylabel("Mean estimate minus population MMD²")
    _ax.legend()
    _fig
    return (bias,)


@app.cell(hide_code=True)
def _(bias):
    mo.md(f"For the equal-size null, the observed mean biases are **{bias[0, 0]:.3f}** "
          f"for U and **{bias[0, 1]:.3f}** for V. The error bars show one Monte Carlo standard error. "
          f"The population MMD² is {bias[0, 4]:.3f} under the null and {bias[2, 4]:.3f} under the shift. "
          "Every trial contributes, and a failed calculation stops the study.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Predict how Monte Carlo variation in the bias comparison changes with more trials, then compare the four scenarios using the same seed as the example.
    """)
    return


@app.cell
def _():
    trials = mo.ui.slider(20, 100, step=20, value=20, label="Trials per scenario")
    trials
    return (trials,)


@app.cell
def _(trials):
    _bias = bias_study(trials=trials.value, seed=123)
    _fig, _ax = plt.subplots()
    _ax.plot(range(4), _bias[:, 0], "o-", label="U")
    _ax.plot(range(4), _bias[:, 1], "o-", label="V")
    _ax.axhline(0, color=INK, linewidth=0.8)
    _ax.set_xticks(range(4), ["null equal", "null unequal", "shift equal", "shift unequal"], rotation=20)
    _ax.set_ylabel("Mean bias")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - Calibration describes how often a procedure rejects across repeated experiments generated under a specified design.
    - Wilson intervals describe Monte Carlo uncertainty in the rejection frequencies estimated from those experiments.
    - U and V estimators handle within-sample diagonal terms differently, which affects their finite-sample bias.
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
    mo.callout(mo.md('This small simulation is a teaching example. See the [assumptions and limits](../reference/resampling.md#assumptions-and-limits) and [evidence](../reference/resampling.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Review permutation tests](permutation-tests.py).
    - [Explore statistical evidence](../validation/index.md).
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


@app.function(hide_code=True)
def calibration(shifts, trials, permutations, seed):
    """Return rejection rates and Wilson intervals without discarding failures."""
    streams = np.random.SeedSequence(seed).spawn(len(shifts) * trials)
    rows = []
    for index, shift in enumerate(shifts):
        rejected = 0
        for trial in range(trials):
            data_seed, test_seed = streams[index * trials + trial].generate_state(2)
            rng = np.random.default_rng(data_seed)
            x = rng.normal(size=(24, 1))
            y = rng.normal(loc=shift, size=(24, 1))
            result = energy_test(x, y, int(test_seed), permutations)
            exceedances = np.count_nonzero(np.asarray(result.null_distribution) >= float(result.statistic))
            rejected += (1 + exceedances) / (1 + permutations) <= 0.05
        rate = rejected / trials
        z = 1.959963984540054
        denominator = 1 + z * z / trials
        center = (rate + z * z / (2 * trials)) / denominator
        half = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
        rows.append((rate, center - half, center + half))
    return np.asarray(rows)


@app.function(hide_code=True)
def bias_study(trials, seed):
    """Compare Gaussian-kernel MMD squared U and V estimates with exact truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for shift, n, m in ((0, 24, 24), (0, 16, 32), (0.75, 24, 24), (0.75, 16, 32)):
        truth = 2 / np.sqrt(3) * (1 - np.exp(-shift**2 / 6))
        estimates = []
        for _ in range(trials):
            x, y = rng.normal(size=n), rng.normal(loc=shift, size=m)
            xx = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
            yy = np.exp(-0.5 * (y[:, None] - y[None, :]) ** 2)
            xy = np.exp(-0.5 * (x[:, None] - y[None, :]) ** 2)
            u = (xx.sum() - np.trace(xx)) / (n * (n - 1)) + (yy.sum() - np.trace(yy)) / (m * (m - 1)) - 2 * xy.mean()
            v = xx.mean() + yy.mean() - 2 * xy.mean()
            estimates.append((u - truth, v - truth))
        values = np.asarray(estimates)
        rows.append((*values.mean(axis=0), *(values.std(axis=0, ddof=1) / np.sqrt(trials)), truth))
    return np.asarray(rows)


@app.cell(hide_code=True)
def _(bias, study):
    def test_study_outputs_are_finite():
        assert np.isfinite(study).all()
        assert np.isfinite(bias).all()
        assert np.all((study[:, 0] >= 0) & (study[:, 0] <= 1))

    return


if __name__ == "__main__":
    app.run()
