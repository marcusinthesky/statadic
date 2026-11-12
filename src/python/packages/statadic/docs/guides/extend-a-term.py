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
"""Implement a training-fitted design term."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from statadic.api.regression import Regression
    from statadic.api.data import Observations

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


    from dataclasses import dataclass
    from typing import override
    from statadic.api.design import Intercept, Term
    from statadic.api.representation import PreparedGeometry
    from statadic.estimation.linear import LeastSquares
    from statadic.api.roles import InputRole
    from statadic.sample.packed import PackedClouds
    from collections.abc import Sequence
    from statadic.api.representation import RowId
    from statadic.estimation.linear import LinearFit, LinearProblem
    from jax import Array


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Extend a design term

    Define a term that learns a training mean and uses it to transform new rows.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    A centered predictor expresses the fitted intercept at the training mean, while its slope describes a change in the original predictor. The fitted term stores that mean and applies it consistently when new rows arrive. For a new observation, the value of the transformed column comes from the difference between its input and the mean of the training sample. This matters when a model serves predictions in batches of different sizes, because the meaning of a coefficient should not depend on the current batch.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Before you start

    For an introduction to design terms and prediction in a fitted model, see the [regression reference](../reference/regression.md).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Fit a centered term

    The current extension hook is the internal `Term._prepare_term` interface, which learns the mean from the training rows and returns a fitted object for new observations. This hook may change. The fitted object also declares its input role and column name, which lets the surrounding design align new data and label the resulting coefficient.
    """)
    return


@app.class_definition
@dataclass(frozen=True)
class FittedCentered:
    """Retain the training mean for all subsequent transformations."""

    column: str
    mean: Array

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Declare the prediction input."""
        return {self.column: "numeric"}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Name the coefficient in the learned design."""
        return (f"centered({self.column})",)

    def transform(self, data: Observations) -> Array:
        """Encode each new row independently of the other prediction rows."""
        return (data.numeric(self.column) - self.mean)[:, None]


@app.class_definition
@dataclass(frozen=True)
class Centered(Term):
    """Center a numeric column around its training sample mean."""

    column: str

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare the training input."""
        return {self.column: "numeric"}

    @override
    def _prepare_term(self, training: Observations) -> FittedCentered:
        """Learn the mean once on the aligned training rows."""
        return FittedCentered(self.column, jnp.mean(training.numeric(self.column)))


@app.cell
def _():
    training = Observations({"age": [20.0, 30.0, 40.0], "y": [1.0, 2.0, 3.0]})
    fit = Regression("y", Intercept() + Centered("age"), estimator=OverdeterminedLeastSquares()).estimate(training).require_valid()
    mo.Html(fit.report().to_html())
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Transform a new row

    A new row is centered using the mean learned from training, so its transformed value is independent of the other rows supplied for prediction.
    """)
    return


@app.cell
def _(fit):
    prediction = fit.predict({"age": [50.0]})
    mo.md(f"Predicted outcome: {float(prediction[0]):.2f}")
    return (prediction,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore

    Before moving the input, predict the fitted outcome at the training mean of 30, where the centered value is zero and the intercept supplies the prediction. The graph marks the selected age on the same fitted line used for prediction.
    """)
    return


@app.cell
def _():
    age = mo.ui.slider(10, 60, value=30, label="Age")
    age
    return (age,)


@app.cell
def _(age, fit):
    _ages = np.linspace(10, 60, 51)
    _predicted = fit.predict({"age": _ages.tolist()})
    _selected = fit.predict({"age": [float(age.value)]})[0]
    _fig, _ax = plt.subplots()
    _ax.plot(_ages, _predicted)
    _ax.scatter([age.value], [float(_selected)], color=BLUE)
    _ax.set_xlabel("Age")
    _ax.set_ylabel("Predicted outcome")
    mo.vstack([mo.md(f"Predicted outcome: {float(_selected):.2f}"), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Result

    The fitted term carries its training mean and names the transformed column, so predictions agree whether a row is supplied alone or in a batch.
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
    mo.callout(mo.md("Read the [regression assumptions](../reference/regression.md#assumptions-and-limits) and [evidence](../reference/regression.md#evidence). The internal preparation hook may change."), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - [Regression](../reference/regression.md)
    - [Ordination](../reference/ordination.md)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class MeanDistance:
    """L1 distance between the mass-weighted means of two empirical measures."""

    def prepare(
        self, measures: PackedClouds, *, row_ids: Sequence[RowId] | None = None
    ) -> PreparedGeometry:
        """Retain the reference collection and its column order."""
        keys = range(measures.n_observations) if row_ids is None else row_ids
        return PreparedGeometry(self, measures, tuple(keys))

    @staticmethod
    def cross(left: PackedClouds, right: PackedClouds) -> Array:
        """Return the left-by-right distances between weighted means."""
        left_means = jnp.sum(left.coordinates * left.masses[..., None], axis=-2)
        right_means = jnp.sum(right.coordinates * right.masses[..., None], axis=-2)
        return jnp.abs(left_means[:, None, :] - right_means[None, :, :]).sum(axis=-1)

    @staticmethod
    def matrix(measures: PackedClouds) -> None:
        """Offer pairwise values without asserting a Hilbertian construction."""
        del measures


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class OverdeterminedLeastSquares:
    """Use statadic least squares only for designs with more rows than columns."""

    @staticmethod
    def estimate(problem: LinearProblem) -> LinearFit:
        """Enforce the downstream policy while preserving the native fit state."""
        n_rows, n_columns = jnp.shape(problem.design)[-2:]
        if n_rows <= n_columns:
            message = "This estimator requires more rows than design columns."
            raise ValueError(message)
        return LeastSquares().estimate(problem)


@app.cell(hide_code=True)
def _(prediction):
    def test_prediction_uses_training_mean():
        assert abs(float(prediction[0]) - 4.0) < 1e-5

    return


if __name__ == "__main__":
    app.run()
