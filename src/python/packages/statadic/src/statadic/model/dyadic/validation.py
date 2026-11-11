# jaxtyping shape strings are runtime contracts.
"""Eager validation of dyadic samples and model/design specifications.

The array predicates are shared component checks for typed dyadic inference
requests; they preserve shape-before-domain error ordering for host labels.
"""

from __future__ import annotations

from collections.abc import (  # runtime hint  # noqa: TC003
    Hashable,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING

import jax.numpy as jnp

from statadic.model.dyadic.specification import DyadicInputError
from statadic.operators.design import (  # noqa: TC001
    ArrayView,
    NodeIndex,
)

__all__: list[str] = []

if TYPE_CHECKING:
    from statadic.model.dyadic.specification import DyadicModel, DyadicSample

MIN_DYADIC_OBSERVATIONS = 2


def _array_shape(value: object) -> tuple[int, ...] | None:
    """Return an array-like's shape, or ``None`` if it is not array-shaped.

    Deliberately dtype-neutral: it must report the shape of a complex or
    string-labelled array so the dimension guards fire before the
    finite-real guard, preserving the retired NumPy ordering of errors.
    """
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and all(isinstance(size, int) for size in shape):
        return shape
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return (len(value),)
    return None


def _is_finite_real_array(value: object) -> bool:
    """Report whether ``value`` is an all-finite real numeric array."""
    if _array_shape(value) is None:
        return False
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return False
    dtype = array.dtype
    return (
        jnp.issubdtype(dtype, jnp.number)
        and not jnp.issubdtype(dtype, jnp.bool_)
        and not jnp.issubdtype(dtype, jnp.complexfloating)
        and bool(jnp.all(jnp.isfinite(array)))
    )


def _validate_endpoint_index[NodeIdT: Hashable](
    endpoint_i: Sequence[NodeIdT],
    endpoint_j: Sequence[NodeIdT],
    node_index: NodeIndex[NodeIdT] | None = None,
) -> NodeIndex[NodeIdT]:
    try:
        if node_index is None:
            return NodeIndex.from_endpoints(endpoint_i, endpoint_j)
        i_position = node_index.positions_for(endpoint_i)
        j_position = node_index.positions_for(endpoint_j)
    except ValueError as error:
        raise DyadicInputError(str(error)) from error
    if _array_shape(endpoint_i) != _array_shape(endpoint_j):
        message = "dyadic endpoint arrays must be aligned one-dimensional vectors"
        raise DyadicInputError(message)
    observed = set(i_position.tolist()) | set(j_position.tolist())
    if observed != set(range(len(node_index))):
        message = "node index must exactly match the observed dyadic endpoint IDs"
        raise DyadicInputError(message)
    if bool(jnp.any(i_position == j_position)):
        message = "undirected dyadic samples cannot contain self-dyads"
        raise DyadicInputError(message)
    return node_index


def _validate_sample[NodeIdT: Hashable](
    sample: DyadicSample[NodeIdT],
    *,
    node_index: NodeIndex[NodeIdT] | None = None,
) -> NodeIndex[NodeIdT]:
    """Validate a nondegenerate real outcome and its distinct entity pairs."""
    if sample.outcome.ndim != 1:
        message = "outcome and endpoint IDs must be aligned one-dimensional arrays"
        raise DyadicInputError(message)
    n_dyads = sample.outcome.shape[0]
    if len(sample.endpoint_i) != n_dyads or len(sample.endpoint_j) != n_dyads:
        message = "outcome and endpoint IDs must be aligned one-dimensional arrays"
        raise DyadicInputError(message)
    if n_dyads < MIN_DYADIC_OBSERVATIONS:
        message = "dyadic estimation requires at least two observed dyads"
        raise DyadicInputError(message)
    if not _is_finite_real_array(sample.outcome):
        message = "dyadic outcomes must be finite real numeric values"
        raise DyadicInputError(message)
    outcome = jnp.asarray(sample.outcome)
    if bool(jnp.all(outcome == outcome[0])):
        message = "dyadic outcomes must be nonconstant so R-squared is defined"
        raise DyadicInputError(message)
    return _validate_endpoint_index(sample.endpoint_i, sample.endpoint_j, node_index)


def _validate_regressors(
    regressor_names: list[str],
    columns: Mapping[str, ArrayView],
    n_dyads: int,
) -> None:
    if not isinstance(regressor_names, list) or any(
        not isinstance(name, str) or not name for name in regressor_names
    ):
        message = "dyadic regressor names must be non-empty strings"
        raise DyadicInputError(message)
    if len(regressor_names) != len(set(regressor_names)):
        message = "dyadic regressor names must be unique"
        raise DyadicInputError(message)
    if "intercept" in regressor_names:
        message = "dyadic regressor name 'intercept' is reserved"
        raise DyadicInputError(message)
    if not isinstance(columns, dict):
        message = "dyadic regressor columns must be supplied by name"
        raise DyadicInputError(message)
    missing = [name for name in regressor_names if name not in columns]
    if missing:
        message = f"missing dyadic regressor columns: {missing}"
        raise DyadicInputError(message)
    misaligned = [
        name for name in regressor_names if _array_shape(columns[name]) != (n_dyads,)
    ]
    if misaligned:
        message = f"misaligned dyadic regressor columns: {misaligned}"
        raise DyadicInputError(message)
    invalid = [
        name for name in regressor_names if not _is_finite_real_array(columns[name])
    ]
    if invalid:
        message = f"dyadic regressors must be finite real numeric values: {invalid}"
        raise DyadicInputError(message)


def _validate_design_inputs[NodeIdT: Hashable](
    regressor_names: list[str],
    columns: Mapping[str, ArrayView],
    endpoint_i: Sequence[NodeIdT],
    endpoint_j: Sequence[NodeIdT],
    *,
    node_index: NodeIndex[NodeIdT] | None = None,
) -> NodeIndex[NodeIdT]:
    """Validate the component-wise public design-builder boundary."""
    index = _validate_endpoint_index(endpoint_i, endpoint_j, node_index)
    endpoint_shape = _array_shape(endpoint_i)
    if endpoint_shape is None or len(endpoint_shape) != 1:
        message = "dyadic endpoint arrays must be aligned one-dimensional vectors"
        raise DyadicInputError(message)
    _validate_regressors(regressor_names, columns, endpoint_shape[0])
    return index


def _validate_model[NodeIdT: Hashable](
    model: DyadicModel[NodeIdT],
) -> NodeIndex[NodeIdT]:
    """Validate one aligned finite regressor specification."""
    _validate_effect_options(
        node_effects=model.node_effects, effect_name_prefix=model.effect_name_prefix
    )
    index = _validate_sample(model.sample)
    _validate_regressors(
        model.regressor_names, model.columns, model.sample.outcome.shape[0]
    )
    return index


def _validate_effect_options(*, node_effects: bool, effect_name_prefix: str) -> None:
    """Validate the structural symmetric node-effect specification."""
    if not isinstance(node_effects, (bool, jnp.bool_)):
        message = "node_effects must be a boolean"
        raise DyadicInputError(message)
    if not isinstance(effect_name_prefix, str) or not effect_name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise DyadicInputError(message)
