"""Retained point-estimation state for undirected dyadic regression."""

from __future__ import annotations

from collections.abc import (
    Hashable,  # runtime hint
)
from dataclasses import dataclass

import jax  # runtime dataclass hint  # noqa: TC002

from statadic.model.dyadic import (  # runtime dataclass hint  # noqa: TC001
    DyadicModel,
    DyadicSample,
)
from statadic.operators.design import (
    NodeIndex,  # runtime dataclass hint  # noqa: TC001
)

__all__ = [
    "DyadicComputationError",
    "DyadicFit",
]


class DyadicComputationError(RuntimeError):
    """Report a failed dyadic fit or bootstrap computation."""


@dataclass(frozen=True)
class DyadicFit[NodeIdT: Hashable = Hashable]:
    """Actual rank-cleaned point fit, with no resampling or inference state.

    ``model`` records the specification as supplied. Later inference reads only
    the owned ``sample``, ``regressor_names`` and ``design``, so mutating a
    caller's arrays after fitting cannot change what is resampled.
    """

    model: DyadicModel[NodeIdT]
    sample: DyadicSample[NodeIdT]
    regressor_names: tuple[str, ...]
    node_index: NodeIndex[NodeIdT]
    coefficients: dict[str, float]
    effect_one_sd: dict[str, float]
    overall_r2: float
    within_r2: float | None
    sse_node_effects_only: float | None
    sse_full: float
    design_rank: int
    dropped_collinear_columns: tuple[str, ...]
    fwl_coefficients: dict[str, float]
    design: jax.Array
    column_names: tuple[str, ...]
    parameters: jax.Array
    fitted: jax.Array
    residual: jax.Array
