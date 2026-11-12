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
"""Compare energy distance and scale."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.representation import Energy
    from statadic.discrepancy.mmd.energy import energy_distance
    from statadic.sample.packed import pack
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
    # Energy distance and scale

    Compare two samples and see how the choice of ground distance changes what the comparison can detect.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    Two laboratories may report clouds of measurements with similar directions but different magnitudes. Angular distance ignores magnitude. Energy distance combines distances between and within the clouds, so its meaning depends on the ground distance used for each pair. By calculating the raw quantity under Euclidean and angular ground, you can see why the same data may answer different questions about direction and magnitude.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    You need familiarity with Python arrays whose rows represent observations, as well as the idea of a distance between two points. The [first regression](../start/first-regression.py) introduces the wider STATADIC workflow.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    We begin with two small samples whose distances can be calculated by hand, so the numerical result has an independent check. A second example draws a reproducible cloud of nonzero vectors and compares it with a scaled copy of the same observations.
    """)
    return


@app.cell
def _():
    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([[2.0], [3.0]])
    return x, y


@app.cell(hide_code=True)
def _(x, y):
    _table = mo.ui.table([
        {"sample": "X", "value": float(x[i, 0])} for i in range(2)
    ] + [
        {"sample": "Y", "value": float(y[i, 0])} for i in range(2)
    ])
    _fig, _ax = plt.subplots()
    _ax.scatter(x[:, 0], jnp.zeros(2), label="X")
    _ax.scatter(y[:, 0], jnp.ones(2), label="Y")
    _ax.set(xlabel="Observation", ylabel="Sample", yticks=[0, 1], yticklabels=["X", "Y"])
    _ax.legend()
    mo.vstack([_table, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Calculate raw energy

    For these samples the mean cross distance is 2 and each within-sample mean is 0.5. The V-statistic includes self-pairs in the within-sample means, so the raw energy for these two small samples is $2(2)-0.5-0.5=3$.
    """)
    return


@app.cell
def _(x, y):
    raw = energy_distance(pack([x]), pack([y]), metric="euclidean")
    mo.plain_text(f"Raw Euclidean V-energy: {raw}")
    return (raw,)


@app.cell(hide_code=True)
def _(raw):
    mo.md(f"The computed value is **{float(raw[0]):.2f}**, matching the calculation above. This is the raw V-energy rather than its square root.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Compare the grounds

    Angular ground compares direction and ignores positive changes of scale, while Euclidean ground also responds to magnitude. Using the same points for both calculations makes the effect of the ground distance clear. For example, vectors $(1,0)$ and $(3,0)$ point in the same direction, but their Euclidean separation is two units.
    """)
    return


@app.cell
def _():
    cloud = make_cloud(seed=42, n=32)
    scaled = 2.0 * cloud
    angular = energy_distance(pack([cloud]), pack([scaled]), metric="angular")
    euclidean = energy_distance(pack([cloud]), pack([scaled]), metric="euclidean")
    mo.plain_text(f"Angular: {angular}\nEuclidean: {euclidean}")
    return angular, cloud, euclidean


@app.cell(hide_code=True)
def _(cloud):
    _fig, _ax = plt.subplots()
    _ax.scatter(cloud[:, 0], cloud[:, 1], label="Original", s=18)
    _ax.scatter(2 * cloud[:, 0], 2 * cloud[:, 1], label="Scaled", s=18, alpha=0.6)
    _ax.set(xlabel="First coordinate", ylabel="Second coordinate", title="A cloud and its scaled copy")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(angular, euclidean):
    mo.md(f"Angular energy is **{float(angular[0]):.5f}**, while Euclidean energy is **{float(euclidean[0]):.3f}**. The first sees directions, and the second sees the change in magnitude.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Compare raw and rooted values

    The host representation uses a rooted V-energy by default, so comparing it
    with the scalar result makes the convention visible before the geometry
    is used in another analysis. These conventions describe the same empirical
    construction on different scales, and their numerical values should be labelled accordingly.
    """)
    return


@app.cell
def _(raw, x, y):
    rooted = Energy(ground="euclidean").cross(pack([x]), pack([y]))
    mo.plain_text(f"Raw: {raw}\nRooted: {rooted}")
    return (rooted,)


@app.cell(hide_code=True)
def _(rooted):
    mo.md(f"The rooted value is **{float(rooted[0, 0]):.3f}**, the square root of the raw value in this example.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Before moving the slider, predict which result will change as the same cloud is stretched away from the origin. The paired construction makes the geometry visible, while an independent two-sample test would require a different sampling design. If scale matters to the scientific question, choose a ground that preserves that distinction before comparing the samples.
    """)
    return


@app.cell
def _():
    scale = mo.ui.slider(0.5, 3.0, step=0.25, value=2.0, label="Scale multiplier")
    scale
    return (scale,)


@app.cell
def _(cloud, scale):
    _scaled = scale.value * cloud
    _a = energy_distance(pack([cloud]), pack([_scaled]), metric="angular")
    _e = energy_distance(pack([cloud]), pack([_scaled]), metric="euclidean")
    mo.md(f"Angular: **{float(_a[0]):.5f}** · Euclidean: **{float(_e[0]):.3f}**")
    return


@app.cell(hide_code=True)
def _(cloud, scale):
    _fig, _ax = plt.subplots()
    _ax.scatter(cloud[:, 0], cloud[:, 1], s=14, label="Original")
    _ax.scatter(scale.value * cloud[:, 0], scale.value * cloud[:, 1], s=14, alpha=0.6, label="Scaled")
    _ax.set(xlabel="First coordinate", ylabel="Second coordinate")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - The scalar API returns raw V-energy with an explicit ground.
    - Angular ground can hide a positive change in scale, so matching directions alone need not imply matching magnitudes.
    - A discrepancy describes separation rather than a p-value, which requires an additional null hypothesis and a suitable calibration.
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
    mo.callout(mo.md('Angular ground requires nonzero vectors. See the [assumptions and limits](../reference/energy.md#assumptions-and-limits) and [evidence](../reference/energy.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Compare distribution-valued observations](distributions.py).
    - [Review distance properties](../reference/catalogue-metrics.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def make_cloud(seed: int, n: int):
    """Draw reproducible nonzero vectors for the scale comparison."""
    return jax.random.normal(jax.random.key(seed), (n, 2)) + jnp.array([2.0, 1.0])


@app.cell(hide_code=True)
def _(angular, euclidean, raw, rooted):
    def test_energy_examples():
        assert abs(float(raw[0]) - 3.0) < 1e-5
        assert abs(float(angular[0])) < 1e-5
        assert float(euclidean[0]) > 0.0
        assert abs(float(rooted[0, 0]) ** 2 - float(raw[0])) < 1e-5

    return


if __name__ == "__main__":
    app.run()
