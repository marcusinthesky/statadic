# jaxtyping shape strings are runtime contracts.
"""Undirected dyadic samples and model specifications.

Endpoint IDs remain host tuples of arbitrary hashable labels. Numeric inputs
are converted at fitting under the caller's JAX precision configuration.
Node effects describe the model; bootstrap controls live in inference.dyadic.
"""

from __future__ import annotations

from collections.abc import (  # runtime hint  # noqa: TC003
    Hashable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass

from statadic.operators.design import (
    ArrayView,  # runtime dataclass hint  # noqa: TC001
    LabelArray,  # noqa: TC001
    node_labels_from_array,
)

__all__ = [
    "DyadicInputError",
    "DyadicModel",
    "DyadicSample",
]


class DyadicInputError(ValueError):
    """Report invalid or incompatible dyadic estimation inputs."""


@dataclass(frozen=True, init=False)
class DyadicSample[NodeIdT: Hashable = Hashable]:
    """Outcome and canonical endpoint IDs for one undirected dyadic panel.

    ``endpoint_i``/``endpoint_j`` retain typed host labels. Use ``from_arrays``
    for external arrays whose Python element type has been erased.
    """

    outcome: ArrayView
    endpoint_i: tuple[NodeIdT, ...]
    endpoint_j: tuple[NodeIdT, ...]

    def __init__(
        self,
        outcome: ArrayView,
        endpoint_i: Sequence[NodeIdT],
        endpoint_j: Sequence[NodeIdT],
    ) -> None:
        """Preserve the numeric input and retain immutable typed node labels."""
        for endpoint in (endpoint_i, endpoint_j):
            if isinstance(endpoint, (str, bytes, bytearray)):
                message = (
                    "dyadic endpoint IDs must be a sequence of labels, not one "
                    "string or bytes value"
                )
                raise DyadicInputError(message)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "endpoint_i", tuple(endpoint_i))
        object.__setattr__(self, "endpoint_j", tuple(endpoint_j))

    @staticmethod
    def from_arrays(
        outcome: ArrayView, endpoint_i: LabelArray, endpoint_j: LabelArray
    ) -> DyadicSample[Hashable]:
        """Decode external label arrays into checked, immutable host tuples."""
        try:
            left = node_labels_from_array(endpoint_i)
            right = node_labels_from_array(endpoint_j)
        except ValueError as error:
            raise DyadicInputError(str(error)) from error
        return DyadicSample(outcome, left, right)


@dataclass(frozen=True)
class DyadicModel[NodeIdT: Hashable = Hashable]:
    """Regressor selection and aligned arrays for one dyadic model."""

    regressor_names: list[str]
    columns: Mapping[str, ArrayView]
    sample: DyadicSample[NodeIdT]
    node_effects: bool = False
    effect_name_prefix: str = "node_effect"
