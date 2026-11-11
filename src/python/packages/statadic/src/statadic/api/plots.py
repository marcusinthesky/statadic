"""Build plot specifications from retained reports and predictors."""

from __future__ import annotations

import math
from collections.abc import (  # noqa: TC003 -- runtime annotations
    Callable,
    Iterable,
    Mapping,
)
from typing import TYPE_CHECKING, SupportsFloat

from statadic.api.regression import LinearResult
from statadic.api.reporting import PlotSpec, ReferenceLine

__all__ = ["prediction_line", "residual_plot"]

if TYPE_CHECKING:
    from statadic.core.typing import Array


def residual_plot(fit: LinearResult) -> PlotSpec:
    """Plot retained response residuals against retained fitted values."""
    if not isinstance(fit, LinearResult):
        message = "residual_plot requires a LinearResult."
        raise TypeError(message)
    fit.require_valid()
    fitted = tuple(float(value) for value in fit.core.fitted_values)
    residuals = tuple(float(value) for value in fit.core.residuals)
    return PlotSpec(
        "scatter",
        fitted,
        residuals,
        xlabel="Fitted values",
        ylabel="Response residuals",
        reference_lines=(ReferenceLine((min(fitted), max(fitted)), (0.0, 0.0)),),
        metadata={"residual_type": "response"},
    )


def prediction_line(
    predict: Callable[[Mapping[str, object]], Array],
    x: str,
    values: Iterable[SupportsFloat],
    *,
    at: Mapping[str, object] | None = None,
    label: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str = "",
) -> PlotSpec:
    """Evaluate a prediction callable on a finite scalar grid.

    Pass a bound ``fit.predict`` method, or bind a prediction kind with
    ``functools.partial``. Grid scalar objects retain their original dtypes.
    """
    if at is not None and x in at:
        message = "at cannot replace the plotted x values"
        raise ValueError(message)
    grid = tuple(values)
    if not grid or any(getattr(value, "ndim", 0) != 0 for value in grid):
        message = "Prediction grid must be a nonempty vector of scalar values."
        raise ValueError(message)
    x_coordinates = tuple(float(value) for value in grid)
    if not all(math.isfinite(value) for value in x_coordinates):
        message = "Prediction grid values must be finite."
        raise ValueError(message)
    data = {x: grid, **(at or {})}
    result = predict(data)
    if result.ndim != 1 or result.shape[0] != len(grid):
        message = "Predictions must be a vector matching the supplied grid length."
        raise ValueError(message)
    coordinates = tuple(float(value) for value in result)
    if not all(math.isfinite(value) for value in coordinates):
        message = "Prediction values must be finite."
        raise ValueError(message)
    return PlotSpec(
        "line",
        x_coordinates,
        coordinates,
        label=label,
        xlabel=x if xlabel is None else xlabel,
        ylabel="" if ylabel is None else ylabel,
        title=title,
    )
