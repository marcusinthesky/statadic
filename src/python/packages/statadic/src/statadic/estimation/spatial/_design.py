# jaxtyping shape strings are runtime contracts.
"""Shared complete-array and spatially lagged design preparation."""

from __future__ import annotations

import jax.numpy as jnp

from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
)
from statadic.estimation._validation import validate_parameter_names
from statadic.operators.spatial import (  # noqa: TC001
    SpatialWeights,
    spatial_lag,
)

_DESIGN_DIMENSIONS = 2


def graph_parameter_names(
    names: tuple[str, ...] | None,
    base: tuple[str, ...] | None,
    slx_columns: tuple[int, ...],
    graph_name: str | None,
    spatial: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Qualify generated graph terms and validate the complete column schema.

    names=None constructs the base/WX/spatial schema for preflight; a fitted
    schema preserves any intervening native endogenous columns.
    """
    if graph_name is None or base is None:
        return names
    if not graph_name:
        message = "graph_name must be a nonempty string."
        raise ValueError(message)
    if any(column < 0 or column >= len(base) for column in slx_columns):
        message = "slx_columns must select explicit design columns."
        raise ValueError(message)
    if names is None:
        names = (*base, *("WX" for _ in slx_columns), *spatial)
    result = list(names)
    for offset, column in enumerate(slx_columns, start=len(base)):
        result[offset] = f"W[{graph_name}]:{base[column]}"
    for offset, name in enumerate(spatial, start=len(result) - len(spatial)):
        result[offset] = f"{name}[{graph_name}]"
    qualified = tuple(result)
    validate_parameter_names(qualified, len(qualified))
    return qualified


def spatial_design(
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialWeights,
    slx_columns: tuple[int, ...],
    parameter_names: tuple[str, ...] | None,
) -> tuple[Array, Array, tuple[str, ...] | None]:
    """Retain complete real arrays and explicitly requested Durbin columns."""
    x, y = jnp.asarray(design), jnp.asarray(outcome)
    dtype = jnp.result_type(x, y, weights.values.dtype, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Spatial regression requires real arrays."
        raise ValueError(message)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(dtype)
    if not jnp.issubdtype(y.dtype, jnp.floating):
        y = y.astype(dtype)
    if (
        x.ndim != _DESIGN_DIMENSIONS
        or y.shape != (x.shape[0],)
        or x.shape[0] != weights.values.shape[0]
    ):
        message = "Spatial design, outcome and W unit rows must align."
        raise ValueError(message)
    validate_parameter_names(parameter_names, x.shape[1])
    if len(set(slx_columns)) != len(slx_columns) or any(
        c < 0 or c >= x.shape[1] for c in slx_columns
    ):
        message = "slx_columns must select unique explicit design columns."
        raise ValueError(message)
    names = parameter_names
    if slx_columns:
        x = jnp.column_stack([x, spatial_lag(weights, x[:, jnp.asarray(slx_columns)])])
        if names is not None:
            names = (*names, *("W:" + names[c] for c in slx_columns))
    return x, y, names
