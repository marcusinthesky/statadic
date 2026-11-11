# jaxtyping shape strings are runtime contracts.
"""Validate dyadic inference controls, node-count schedules and fitted designs."""

from __future__ import annotations

from collections.abc import (
    Hashable,  # runtime hint
)
from typing import TYPE_CHECKING

import jax.numpy as jnp

from statadic.inference.dyadic.contracts import NodeCountSchedule
from statadic.model.dyadic import DyadicInputError
from statadic.model.dyadic.validation import _array_shape, _is_finite_real_array

if TYPE_CHECKING:
    from statadic.inference.dyadic.contracts import DyadicBootstrapDesign, DyadicInference
    from statadic.operators.design import (
        NodeIndex,
    )

MATRIX_DIMENSIONS = 2
_INTEGER_TYPES = (int, jnp.integer)
_BOOLEAN_TYPES = (bool, jnp.bool_)


def validate_inference[NodeIdT: Hashable](
    inference: DyadicInference[NodeIdT],
) -> None:
    """Validate eager bootstrap controls before JAX or linear algebra sees them."""
    if (
        not isinstance(inference.bootstrap_iters, _INTEGER_TYPES)
        or isinstance(inference.bootstrap_iters, _BOOLEAN_TYPES)
        or inference.bootstrap_iters < 1
    ):
        message = "node bootstrap requires a positive integer draw count"
        raise DyadicInputError(message)
    if not isinstance(inference.bootstrap_seed, _INTEGER_TYPES) or isinstance(
        inference.bootstrap_seed, _BOOLEAN_TYPES
    ):
        message = "node bootstrap seed must be an integer"
        raise DyadicInputError(message)
    required = inference.required_bootstrap_columns
    if not isinstance(required, tuple) or any(
        not isinstance(name, str) or not name for name in required
    ):
        message = "required bootstrap columns must be non-empty string names"
        raise DyadicInputError(message)
    if len(required) != len(set(required)):
        message = "required bootstrap columns must be unique"
        raise DyadicInputError(message)


def validate_node_schedule[NodeIdT: Hashable](
    schedule: NodeCountSchedule[NodeIdT],
    node_index: NodeIndex[NodeIdT],
    *,
    expected_draws: int | None = None,
) -> None:
    """Validate a multinomial schedule and its exact entity-column mapping."""
    if not isinstance(schedule, NodeCountSchedule):
        message = "node bootstrap requires a NodeCountSchedule"
        raise DyadicInputError(message)
    if schedule.node_index != node_index:
        message = "node-count schedule IDs do not match the dyadic sample"
        raise DyadicInputError(message)
    node_counts = schedule.counts
    n_nodes = len(node_index)
    counts_shape = _array_shape(node_counts)
    if counts_shape is None:
        message = "node counts must be supplied as a two-dimensional count matrix"
        raise DyadicInputError(message)
    if len(counts_shape) != MATRIX_DIMENSIONS or counts_shape[1] != n_nodes:
        width = counts_shape[1] if len(counts_shape) == MATRIX_DIMENSIONS else None
        message = f"node-count matrix has {width} columns for {n_nodes} nodes"
        raise DyadicInputError(message)
    if counts_shape[0] < 1:
        message = "node-count matrix must contain at least one draw"
        raise DyadicInputError(message)
    if expected_draws is not None and counts_shape[0] != expected_draws:
        message = (
            f"node-count matrix has {counts_shape[0]} draws, expected {expected_draws}"
        )
        raise DyadicInputError(message)
    if not _is_finite_real_array(node_counts):
        message = "node counts must be finite nonnegative integers"
        raise DyadicInputError(message)
    counts = jnp.asarray(node_counts)
    if bool(jnp.any(counts < 0)) or bool(jnp.any(counts != jnp.floor(counts))):
        message = "node counts must be finite nonnegative integers"
        raise DyadicInputError(message)
    if bool(jnp.any(jnp.sum(counts, axis=1) != n_nodes)):
        message = f"each node-count draw must sum to the {n_nodes} sampled nodes"
        raise DyadicInputError(message)


def _validate_bootstrap_matrix_and_names[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
) -> list[str]:
    """Validate the numeric matrix and its injective column namespace."""
    design = inputs.design
    column_names = inputs.column_names
    if not isinstance(column_names, list) or not column_names:
        message = "dyadic design column names must be a non-empty list"
        raise DyadicInputError(message)
    outcome_shape = _array_shape(inputs.sample.outcome)
    if outcome_shape is None or _array_shape(design) != (
        outcome_shape[0],
        len(column_names),
    ):
        message = "design shape does not match the dyadic sample and column names"
        raise DyadicInputError(message)
    if not _is_finite_real_array(design):
        message = "dyadic bootstrap design must contain finite real numeric values"
        raise DyadicInputError(message)
    if any(not isinstance(name, str) or not name for name in column_names):
        message = "dyadic design column names must be non-empty strings"
        raise DyadicInputError(message)
    if len(column_names) != len(set(column_names)):
        message = "dyadic design column names must be unique"
        raise DyadicInputError(message)
    return column_names


def _validate_bootstrap_configuration[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
    column_names: list[str],
) -> None:
    """Validate bootstrap flags and coefficient-retention requirements."""
    if not isinstance(inputs.node_effects, _BOOLEAN_TYPES):
        message = "node_effects must be a boolean"
        raise DyadicInputError(message)
    if not isinstance(inputs.effect_name_prefix, str) or not inputs.effect_name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise DyadicInputError(message)
    required = inputs.required_columns
    if not isinstance(required, tuple) or any(
        not isinstance(name, str) or not name for name in required
    ):
        message = "required bootstrap columns must be non-empty string names"
        raise DyadicInputError(message)
    if len(required) != len(set(required)):
        message = "required bootstrap columns must be unique"
        raise DyadicInputError(message)
    missing_required = [name for name in required if name not in column_names]
    if missing_required:
        message = f"required bootstrap columns are absent: {missing_required}"
        raise DyadicInputError(message)
    if inputs.node_effects:
        effect_prefix = f"{inputs.effect_name_prefix}["
        if "intercept" not in column_names or not any(
            name.startswith(effect_prefix) for name in column_names
        ):
            message = (
                "node-effect bootstrap designs require intercept and effect columns"
            )
            raise DyadicInputError(message)


def validate_bootstrap_design[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
    node_index: NodeIndex[NodeIdT],
) -> None:
    """Validate a complete public bootstrap-design request."""
    column_names = _validate_bootstrap_matrix_and_names(inputs)
    _validate_bootstrap_configuration(inputs, column_names)
    validate_node_schedule(inputs.node_schedule, node_index)
