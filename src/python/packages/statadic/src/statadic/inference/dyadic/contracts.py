# jaxtyping shape strings are runtime contracts.
"""Requests, node multiplicities and reports for dyadic bootstrap inference.

Results retain JAX arrays; application reporting edges own host conversions.
NodeCountSchedule stores entity counts, never row-resampling indices.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,  # runtime hint
)
from dataclasses import dataclass, field

import jax  # runtime dataclass hint  # noqa: TC002
import jax.numpy as jnp

from statadic.core.typing import ArrayLike  # noqa: TC001
from statadic.model.dyadic import DyadicModel, DyadicSample  # noqa: TC001
from statadic.operators.design import (  # noqa: TC001
    ArrayView,
    NodeIndex,
)

__all__ = [
    "BootstrapCoefficientSummary",
    "DyadicBootstrapDesign",
    "DyadicBootstrapResult",
    "DyadicFitRequest",
    "DyadicFitResult",
    "DyadicInference",
    "NodeCountSchedule",
]

DEFAULT_BOOTSTRAP_ITERS = 1_999
DEFAULT_BOOTSTRAP_SEED = 42
MIN_ACTIVE_DYADS = 3


@dataclass(frozen=True, init=False)
class NodeCountSchedule[NodeIdT: Hashable = Hashable]:
    """Multinomial counts tied to the exact entity-to-column mapping they use."""

    node_index: NodeIndex[NodeIdT]
    counts: jax.Array

    def __init__(self, node_index: NodeIndex[NodeIdT], counts: ArrayLike) -> None:
        """Retain immutable array counts without silently truncating their dtype."""
        object.__setattr__(self, "node_index", node_index)
        object.__setattr__(self, "counts", jnp.asarray(counts))


@dataclass(frozen=True)
class DyadicInference[NodeIdT: Hashable = Hashable]:
    """Multinomial node-bootstrap controls; structural effects belong to the model."""

    bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    node_schedule: NodeCountSchedule[NodeIdT] | None = None
    required_bootstrap_columns: tuple[str, ...] = ()


def _default_inference[NodeIdT: Hashable]() -> DyadicInference[NodeIdT]:
    """Specialize the schedule-free default to the fit carrier's node ID."""
    return DyadicInference[NodeIdT]()


@dataclass(frozen=True)
class DyadicFitRequest[NodeIdT: Hashable = Hashable]:
    """Complete point-estimation and inference request."""

    model: DyadicModel[NodeIdT]
    inference: DyadicInference[NodeIdT] = field(default_factory=_default_inference)


@dataclass(frozen=True)
class DyadicBootstrapDesign[NodeIdT: Hashable = Hashable]:
    """Rank-cleaned design and shared node-bootstrap schedule."""

    design: ArrayView
    column_names: list[str]
    sample: DyadicSample[NodeIdT]
    node_schedule: NodeCountSchedule[NodeIdT]
    node_effects: bool = False
    effect_name_prefix: str = "node_effect"
    required_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DyadicBootstrapResult:
    """Array-valued coefficient draws and per-draw validity diagnostics."""

    draw_id: jax.Array
    coefficients: dict[str, jax.Array]
    sse: jax.Array
    overall_r2: jax.Array
    within_r2: jax.Array
    active_nodes: jax.Array
    active_dyads: jax.Array
    effective_columns: jax.Array
    effective_rank: jax.Array
    solve_ok: jax.Array

    def coefficient(self, name: str) -> jax.Array:
        """Return one aligned coefficient series, or all-``NaN`` if absent."""
        values = self.coefficients.get(name)
        if values is not None:
            return values
        return jnp.full(self.draw_id.shape, jnp.nan, dtype=self.sse.dtype)


@dataclass(frozen=True)
class BootstrapCoefficientSummary:
    """Named uncertainty and percentile limits for one bootstrap coefficient."""

    standard_error: float
    t_statistic: float
    pvalue: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class DyadicFitResult:
    """Point fit, node-bootstrap draws, and design diagnostics."""

    n_dyads: int
    regressor_names: tuple[str, ...]
    coefficients: dict[str, float]
    effect_one_sd: dict[str, float]
    bootstrap_inference: dict[str, BootstrapCoefficientSummary]
    bootstrap_draws: DyadicBootstrapResult
    bootstrap_valid_draws: int
    bootstrap_requested_draws: int
    bootstrap_seed: int
    overall_r2: float
    within_r2: float | None
    sse_node_effects_only: float | None
    sse_full: float
    design_rank: int
    design_columns: int
    dropped_collinear_columns: tuple[str, ...]
    fwl_coefficients: dict[str, float]
    point_design: jax.Array
    point_design_names: tuple[str, ...]
    point_fitted: jax.Array
    point_residual: jax.Array
