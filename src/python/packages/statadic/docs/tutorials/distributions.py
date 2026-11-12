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
"""Analyze distribution-valued observations."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.comparison import PermutationPlan
    from statadic.api.data import MeasureColumn, Observations
    from statadic.api.design import Categorical, DistanceToLandmarks, Intercept, Linear
    from statadic.api.hypotheses import Hypothesis
    from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
    from statadic.api.regression import GeometricResponse, Reduced, Regression, ResponsePermutation
    from statadic.api.representation import MMD
    from statadic.geometry.ground.similarities import RBFKernel
    from statadic.sample.packed import PackedClouds
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
    # One distribution per observation

    Prepare measures for whole units, map their similarities and predict outcomes for held-out units.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A measurement may itself be a distribution, such as the range of readings collected from one location over a day, where the entire location is one observation containing several atoms. You will fit a geometry on training units, use named landmarks in a regression, and predict new units without rebuilding that geometry or letting information from the new units influence the fitted landmarks.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    You should know how to fit a [first regression](../start/first-regression.py) and how a collection of weighted points represents an empirical distribution. The tutorial uses a fixed kernel, so prior experience with choosing one is unnecessary.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data

    The seeded generator creates 24 units with weighted atoms, an outcome and a scalar price. We train on 18 whole units and reserve six for prediction, preserving their identities.
    """)
    return


@app.cell
def _():
    data = make_observations(seed=7, spread=1.0)
    training = data.take(tuple(range(18)))
    held_out = data.take(tuple(range(18, 24)))
    mo.plain_text(f"Training units: {len(training.row_ids)}; held out: {len(held_out.row_ids)}")
    return held_out, training


@app.cell(hide_code=True)
def _(training):
    _m = training.measure("distribution")
    _table = mo.ui.table([{"unit": training.row_ids[i], "outcome": float(training.numeric("outcome")[i])} for i in range(5)])
    _fig, _ax = plt.subplots()
    for _i in range(3):
        _n = int(_m.counts[_i])
        _ax.scatter(_m.data[_i, :_n, 0], jnp.full(_n, _i))
    _ax.set(xlabel="Atom location", ylabel="Training unit")
    mo.vstack([_table, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Prepare the training geometry

    A fixed RBF kernel defines the comparison between measures, with a bandwidth chosen before examining the outcomes. Preparing it on the training observations preserves their order and supplies the distances used by the ordination.
    """)
    return


@app.cell
def _(training):
    geometry = MMD(kernel=RBFKernel(bandwidth=1.0))
    prepared = geometry.prepare(training.measure("distribution"), row_ids=training.row_ids)
    embedding = Ordination(DistanceGram(), PrincipalCoordinates(dim=2)).estimate(prepared)
    mo.plain_text(f"Training coordinates: {embedding.coordinates.shape}")
    return embedding, geometry, prepared


@app.cell(hide_code=True)
def _(embedding):
    _fig, _ax = plt.subplots()
    _ax.scatter(embedding.coordinates[:, 0], embedding.coordinates[:, 1], s=22)
    _ax.set(xlabel="Coordinate 1", ylabel="Coordinate 2", title="Training units in the measure geometry")
    _ax
    return


@app.cell(hide_code=True)
def _(embedding):
    mo.md(f"The map has **{embedding.coordinates.shape[0]} units** and **{embedding.coordinates.shape[1]} coordinates**. Each point represents a whole distribution, rather than one of its atoms.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Fit and predict with landmarks

    Distances to two named training units become regression terms alongside the scalar price, giving each unit a set of features derived from its distribution. The fitted result retains those training landmarks when it predicts the held-out units, so prediction uses the same geometry. Selecting a kernel or landmarks after looking at held-out outcomes would turn this split into a validation set rather than an independent final evaluation.
    """)
    return


@app.cell
def _(geometry, held_out, training):
    terms = Intercept() + Linear("price") + DistanceToLandmarks(
        "distribution", geometry=geometry, landmarks=("unit-1", "unit-10")
    )
    fit = Regression(response="outcome", terms=terms).estimate(training).require_valid()
    predictions = fit.predict(held_out)
    mo.plain_text(f"Predictions: {predictions}")
    return (predictions,)


@app.cell(hide_code=True)
def _(held_out, predictions):
    _actual = held_out.numeric("outcome")
    _fig, _ax = plt.subplots()
    _ax.scatter(_actual, predictions, s=28)
    _ax.set(xlabel="Observed outcome", ylabel="Predicted outcome", title="Six held-out units")
    _ax
    return


@app.cell(hide_code=True)
def _(held_out, predictions):
    _error = jnp.sqrt(jnp.mean((held_out.numeric("outcome") - predictions) ** 2))
    mo.md(f"The six held-out predictions have an exploratory RMSE of **{float(_error):.3f}**. This small synthetic split illustrates the workflow rather than establishing predictive performance.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Compare groups of units

    A geometric response compares groups using the same prepared distances, while a separate permutation request supplies the reference distribution. Inspect whether any draws failed before interpreting the result, and remember that the small permutation budget here demonstrates the workflow. A real comparison also needs a reason to regard the group labels as exchangeable under its null hypothesis.
    """)
    return


@app.cell
def _(prepared, training):
    grouped = Regression(
        GeometricResponse(prepared, distance_power=1),
        Intercept() + Categorical("group"),
    ).estimate(training)
    group_test = grouped.infer(
        resampling=ResponsePermutation(PermutationPlan(n=31, seed=72))
    ).test(Hypothesis(Reduced(("intercept",))))
    mo.plain_text(group_test.report().to_text())
    return (group_test,)


@app.cell(hide_code=True)
def _(group_test):
    mo.md(f"The group comparison reports **{group_test.evidence.failed} failed draws**. "
          "The outer units, rather than their atoms, are the observations in this permutation.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Before moving the control, predict how a wider distribution might change the distance between two units when their locations stay close. The same seed and kernel are used throughout, so the displayed change comes from the spread of the atoms within each whole observation.
    """)
    return


@app.cell
def _():
    spread = mo.ui.slider(0.5, 2.0, step=0.25, value=1.0, label="Within-unit spread")
    spread
    return (spread,)


@app.cell
def _(spread):
    _data = make_observations(seed=7, spread=spread.value)
    _prepared = MMD(kernel=RBFKernel(bandwidth=1.0)).prepare(
        _data.measure("distribution"), row_ids=_data.row_ids
    )
    _distance = _prepared.cross(_prepared.reference)[0, 1]
    mo.md(f"The first two units have MMD distance **{float(_distance):.3f}** at spread **{spread.value:.2f}**.")
    return


@app.cell(hide_code=True)
def _(spread):
    explore_figure(spread)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you learned

    - A row represents a whole empirical measure, with atoms and masses inside it.
    - Geometry prepared on the training units supplies a map and the features measured relative to selected landmarks.
    - Prediction reuses training landmarks for held-out units, preserving the separation between fitting and evaluation.
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
    mo.callout(mo.md('A selected kernel determines which differences are visible. See the [assumptions and limits](../reference/energy.md#assumptions-and-limits) and [evidence](../reference/energy.md#evidence).'), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps

    - [Explore energy and scale](energy-and-scale.py).
    - [Read the ordination reference](../reference/ordination.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function(hide_code=True)
def explore_figure(spread):
    _data = make_observations(seed=7, spread=spread.value)
    _measures = _data.measure("distribution")
    _fig, _ax = plt.subplots()
    for _i in range(2):
        _n = int(_measures.counts[_i])
        _ax.scatter(_measures.data[_i, :_n, 0], jnp.full(_n, _i), s=150 * _measures.masses[_i, :_n])
    _ax.set(xlabel="Atom location", ylabel="Unit", yticks=[0, 1])
    return _fig


@app.function(hide_code=True)
def make_observations(seed: int, spread: float) -> Observations:
    """Create weighted measures and outcomes for 24 reproducible units."""
    count = 24
    sizes = [3 + i % 4 for i in range(count)]
    shifts = 0.02 * jax.random.normal(jax.random.key(seed), (count,))
    clouds = [
        (jnp.linspace(-1.0, 1.0, size) * spread * (0.5 + 0.1 * (i % 3))
         + 0.15 * i + shifts[i])[:, None]
        for i, size in enumerate(sizes)
    ]
    masses = [jnp.arange(1.0, size + 1.0) for size in sizes]
    index = jnp.arange(float(count))
    row_ids = tuple(f"unit-{i}" for i in range(count))
    measures = PackedClouds.from_clouds(clouds, masses=masses, normalize=True)
    return Observations(
        {"price": jnp.cos(index * 0.7), "outcome": 2.0 + index * 0.2 + jnp.sin(index * 0.9)},
        row_ids=row_ids,
        labels={"group": tuple("a" if i % 2 else "b" for i in range(count))},
        measures={"distribution": MeasureColumn(row_ids, measures)},
    )


@app.cell(hide_code=True)
def _(embedding, group_test, predictions):
    def test_held_out_predictions_and_map():
        assert predictions.shape == (6,)
        assert bool(jnp.all(jnp.isfinite(predictions)))
        assert embedding.coordinates.shape == (18, 2)
        assert group_test.available
        assert group_test.evidence.failed == 0

    return


if __name__ == "__main__":
    app.run()
