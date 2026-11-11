"""Publication helpers consume retained quantities and fitted predictions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_predictor
from statadic.api.base import InvalidFitError
from statadic.api.data import RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.plots import prediction_line, residual_plot
from statadic.api.regression import Regression
from statadic.api.reporting import ParameterRow, Report

from .._typing import check_types, ill_typed
from ._models import poisson_model

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.parametrize("weighted", [False, True])
def test_residual_plot_uses_retained_linear_values(*, weighted: bool) -> None:
    """OLS and WLS use their retained response residuals without refitting."""
    terms = Intercept() + Linear("x")
    model = (
        Regression("y", terms, precision=RelativePrecision("w"))
        if weighted
        else Regression("y", terms)
    )
    fit = model.estimate(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 4.0, 5.0],
            "w": [1.0, 2.0, 1.0, 2.0],
        }
    )
    plot = residual_plot(fit)
    assert plot.kind == "scatter"
    np.testing.assert_allclose(plot.x, fit.core.fitted_values)
    np.testing.assert_allclose(plot.y, fit.core.residuals)
    assert plot.reference_lines[0].y == (0.0, 0.0)
    assert plot.xlabel == "Fitted values"
    assert plot.ylabel == "Response residuals"
    assert plot.metadata["residual_type"] == "response"


@pytest.mark.component
@pytest.mark.contract
def test_residual_plot_rejects_unsupported_and_invalid_fits() -> None:
    """Only valid linear results are accepted."""
    data = {"x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]}
    terms = Intercept() + Linear("x")
    with pytest.raises(InvalidFitError):
        residual_plot(Regression("y", terms).estimate(data))
    with pytest.raises(TypeError, match="LinearResult"):
        residual_plot(poisson_model("y", terms).estimate(data))  # type: ignore[arg-type]


@pytest.mark.component
@pytest.mark.contract
def test_coefficient_plot_uses_retained_intervals() -> None:
    """Intervals and their interpretation come from the retained report."""
    report = Report(
        "fit",
        parameters=(ParameterRow("slope", 0.5, ci_lower=0.1, ci_upper=0.9),),
        metadata={"confidence": 0.95, "reference": "normal", "covariance": "HC1"},
    )
    plot = report.coefficient_plot(reference_value=1.0)
    assert plot.y == (0.5,)
    assert plot.lower == (0.1,)
    assert plot.upper == (0.9,)
    assert plot.labels == ("slope",)
    assert plot.reference_lines[0].y == (1.0, 1.0)
    assert plot.interval_meaning == "95% intervals; normal reference; covariance HC1"
    with pytest.raises(ValueError, match="confidence"):
        Report("fit", parameters=report.parameters).coefficient_plot()
    with pytest.raises(ValueError, match="intervals"):
        Report(
            "fit", parameters=(ParameterRow("x", 1.0),), metadata={"confidence": 0.95}
        ).coefficient_plot()


@pytest.mark.integration
def test_prediction_line_live_and_reloaded(tmp_path: Path) -> None:
    """Live and restored predictors evaluate the same supplied grid."""
    fit = Regression("y", Intercept() + Linear("x")).estimate(
        {"x": jnp.arange(4.0), "y": jnp.array([1.0, 3.0, 5.0, 7.0])}
    )
    path = tmp_path / "predictor"
    fit.save_predictor(path)
    reloaded = load_predictor(path)
    grid = [1.0, 2.0, 3.0]
    live = prediction_line(fit.predict, "x", grid)
    restored = prediction_line(reloaded.predict, "x", grid)
    assert live.x == restored.x
    assert live.y == restored.y
    np.testing.assert_allclose(live.y, fit.predict({"x": grid}))
    np.testing.assert_allclose(
        prediction_line(fit.predict, "x", jnp.asarray(grid)).y, live.y
    )


@pytest.mark.unit
@pytest.mark.contract
def test_prediction_line_rejects_nonfinite() -> None:
    """A nonfinite prediction is rejected before rendering."""

    class Predictor:
        def predict(self, data: object) -> jnp.ndarray:
            del data
            return jnp.array([float("nan")])

    with pytest.raises(ValueError, match="finite"):
        prediction_line(Predictor().predict, "x", [1.0])


@pytest.mark.unit
@pytest.mark.contract
def test_prediction_line_retains_grid_without_x64() -> None:
    """Presentation preserves the supplied host coordinates at either precision."""

    class Predictor:
        def predict(self, data: object) -> jnp.ndarray:
            del data
            return jnp.array([1.0, 2.0])

    grid = np.array([0.1, 0.2], dtype=np.float64)
    with jax.enable_x64(new_val=False):
        plot = prediction_line(Predictor().predict, "x", grid)
    assert plot.x == tuple(float(value) for value in grid)


@pytest.mark.integration
@pytest.mark.contract
def test_prediction_line_accepts_bound_prediction_kind() -> None:
    """A family prediction scale is ordinary callable composition."""
    fit = poisson_model("y", Intercept() + Linear("x")).estimate(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 5.0]}
    )
    predict = partial(fit.predict, kind="link")
    grid = [0.5, 1.5]
    plot = prediction_line(predict, "x", grid)
    np.testing.assert_allclose(plot.y, predict({"x": grid}))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("grid", [[], [float("nan")], [[1.0]]])
def test_prediction_line_rejects_invalid_grid(grid: list[object]) -> None:
    """The presentation boundary requires finite scalar coordinates."""

    def predict(data: object) -> jnp.ndarray:
        del data
        return jnp.ones(1)

    with pytest.raises((TypeError, ValueError)):
        prediction_line(predict, "x", ill_typed(grid))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("result", [jnp.ones(()), jnp.ones((2, 1)), jnp.ones(3)])
def test_prediction_line_rejects_misaligned_predictions(result: jnp.ndarray) -> None:
    """Prediction results retain exactly one value per supplied coordinate."""

    def predict(data: object) -> jnp.ndarray:
        del data
        return result

    with pytest.raises(ValueError, match=r"matching.*grid"):
        prediction_line(predict, "x", [1.0, 2.0])


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.typing
def test_plotting_requires_a_prediction_callable(tmp_path: Path) -> None:
    """Fitted objects and unsupported plot-level kind arguments are rejected."""
    check_types(
        """
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.plots import prediction_line
fit = Regression('y', Intercept() + Linear('x')).estimate(
    {'x': [0.0, 1.0], 'y': [1.0, 2.0]}
)
prediction_line(fit.predict, "x", [1.])
prediction_line(fit, "x", [1.])  # E: not assignable
prediction_line(fit.predict, "x", [1.], kind="mean")  # E: Unexpected keyword
""",
        "prediction_callable_fixture.py",
        tmp_path,
    )
