# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Explicit spatial weights and transformable dense/sparse operator algebra.

Preparation is eager and never changes the graph implicitly. Numerical products
and solves accept fixed-shape JAX arrays or BCOO storage. Sparse storage alone
does not establish a performance guarantee, and is never silently densified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Float, Int  # noqa: TC002

from statadic.core.results import (
    SolverResult,  # noqa: TC001
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.optimize.backends import iterative_linear_solve

__all__ = [
    "ObservedSpatialPeriod",
    "SpatialPanelWeights",
    "SpatialWeights",
    "prepare_spatial_panel_weights",
    "prepare_spatial_weights",
    "spatial_lag",
    "spatial_lags",
    "spatial_logdet",
    "spatial_panel_lag",
    "spatial_solve",
]


_MATRIX_DIMENSIONS = 2
_PANEL_DIMENSIONS = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialWeights[StorageT: Array | BCOO = Array | BCOO]:
    """Aligned weights, numeric unit keys and conservative stability support.

    Unit keys are dynamic integer array leaves; external string labels belong
    to the caller's data adapter. ``limit`` is the reciprocal infinity norm,
    capped at one, so ``abs(rho) < limit`` guarantees invertibility without
    symmetry assumptions. Normalization and isolated rows are visible in the
    retained values and row sums. The zero operator has limit one.
    """

    values: StorageT
    unit_ids: Int[Array, "*batch units"]
    row_sums: Float[Array, "*batch units"]
    limit: Float[Array, "*batch"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialPanelWeights:
    """Fixed or time-varying graph with explicit ordered unit and period keys."""

    values: Array
    unit_ids: Int[Array, "*batch units"]
    period_ids: Int[Array, "*batch periods"]
    limit: Float[Array, "*batch"]


def prepare_spatial_panel_weights(
    values: ArrayLike,
    *,
    period_ids: ArrayLike,
    unit_ids: ArrayLike | None = None,
    normalize_rows: bool = False,
    allow_isolates: bool = False,
) -> SpatialPanelWeights:
    """Prepare exact dense panel weights without changing unit/period alignment.

    ``values`` is (N,N) or (T,N,N). ``unit_ids`` may be (N,) or (T,N); the
    latter must agree in every period, otherwise the caller must reindex before
    fitting. Integer period keys must be strictly increasing. Sparse products
    use ``spatial_panel_lag`` directly; sparse likelihoods remain deferred.
    """
    matrix = jnp.asarray(values)
    periods = jnp.asarray(period_ids)
    if (
        periods.ndim != 1
        or not periods.size
        or not jnp.issubdtype(periods.dtype, jnp.integer)
        or jnp.any(jnp.diff(periods) <= 0)
    ):
        message = "period_ids must be strictly increasing integer keys."
        raise ValueError(message)
    if matrix.ndim not in (_MATRIX_DIMENSIONS, _PANEL_DIMENSIONS) or (
        matrix.ndim == _PANEL_DIMENSIONS and matrix.shape[0] != len(periods)
    ):
        message = "Use fixed (N,N) or aligned time-varying (T,N,N) weights."
        raise ValueError(message)
    keys = None if unit_ids is None else jnp.asarray(unit_ids)
    if keys is not None and keys.ndim == _MATRIX_DIMENSIONS:
        if keys.shape != (len(periods), matrix.shape[-1]) or not jnp.all(
            keys == keys[0]
        ):
            message = (
                "Unit ordering must be identical in every period; reindex explicitly."
            )
            raise ValueError(message)
        keys = keys[0]
    graphs = [
        prepare_spatial_weights(
            matrix if matrix.ndim == _MATRIX_DIMENSIONS else matrix[t],
            unit_ids=keys,
            normalize_rows=normalize_rows,
            allow_isolates=allow_isolates,
        )
        for t in range(1 if matrix.ndim == _MATRIX_DIMENSIONS else len(periods))
    ]
    prepared = (
        graphs[0].values
        if matrix.ndim == _MATRIX_DIMENSIONS
        else jnp.stack([g.values for g in graphs])
    )
    return SpatialPanelWeights(
        prepared,
        graphs[0].unit_ids,
        jnp.asarray(periods),
        jnp.min(jnp.stack([g.limit for g in graphs])),
    )


def _matrix_state(
    matrix: Array | BCOO,
) -> tuple[jnp.ndarray, jnp.ndarray, Array, Array]:
    """Inspect dense or canonical sparse storage without hidden densification."""
    if isinstance(matrix, BCOO):
        if matrix.n_batch or matrix.n_dense or not matrix.unique_indices:
            message = "Use a canonical unbatched BCOO with unique scalar entries."
            raise TypeError(message)
        data, indices = jnp.asarray(matrix.data), jnp.asarray(matrix.indices)
        if jnp.any(indices < 0) or jnp.any(indices >= matrix.shape[0]):
            message = "Sparse weights contain invalid indices."
            raise TypeError(message)
        diagonal = data[indices[:, 0] == indices[:, 1]]
        row_sums = (
            jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
            .at[matrix.indices[:, 0]]
            .add(matrix.data)
        )
        row_norm = (
            jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
            .at[matrix.indices[:, 0]]
            .add(jnp.abs(matrix.data))
        )
        return data, diagonal, row_sums, row_norm
    return (
        jnp.asarray(matrix),
        jnp.diag(jnp.asarray(matrix)),
        jnp.sum(matrix, axis=1),
        jnp.sum(jnp.abs(matrix), axis=1),
    )


@overload
def _normalize_rows(
    matrix: Array, row_sums: Array, row_norm: Array
) -> tuple[Array, Array, Array]: ...


@overload
def _normalize_rows(
    matrix: BCOO, row_sums: Array, row_norm: Array
) -> tuple[BCOO, Array, Array]: ...


def _normalize_rows(
    matrix: Array | BCOO, row_sums: Array, row_norm: Array
) -> tuple[Array | BCOO, Array, Array]:
    """Normalize explicitly requested positive-sum rows, retaining isolated zeros."""
    if jnp.any((jnp.asarray(row_sums) <= 0) & (jnp.asarray(row_norm) != 0)):
        message = "Row normalization requires positive nonzero row sums."
        raise ValueError(message)
    divisor = jnp.where(row_sums != 0, row_sums, 1)
    if isinstance(matrix, BCOO):
        matrix = BCOO(
            (matrix.data / divisor[matrix.indices[:, 0]], matrix.indices),
            shape=matrix.shape,
            indices_sorted=matrix.indices_sorted,
            unique_indices=True,
        )
    else:
        matrix = matrix / divisor[:, None]
    return matrix, row_sums / divisor, row_norm / divisor


@overload
def prepare_spatial_weights(
    values: BCOO,
    *,
    unit_ids: ArrayLike | None = None,
    normalize_rows: bool = False,
    allow_isolates: bool = False,
    allow_diagonal: bool = False,
) -> SpatialWeights[BCOO]: ...


@overload
def prepare_spatial_weights(
    values: ArrayLike,
    *,
    unit_ids: ArrayLike | None = None,
    normalize_rows: bool = False,
    allow_isolates: bool = False,
    allow_diagonal: bool = False,
) -> SpatialWeights[Array]: ...


def prepare_spatial_weights(
    values: ArrayLike | BCOO,
    *,
    unit_ids: ArrayLike | None = None,
    normalize_rows: bool = False,
    allow_isolates: bool = False,
    allow_diagonal: bool = False,
) -> SpatialWeights:
    """Validate one graph eagerly; preserve sparse storage and unit order.

    Sparse inputs must already have canonical unique indices; conversion from
    SciPy sparse matrices belongs to the caller's I/O boundary. Negative edges
    are permitted, but row normalization requires positive nonzero row sums.
    Zero rows require ``allow_isolates=True`` and remain zero when normalized.
    """
    matrix = values if isinstance(values, BCOO) else jnp.asarray(values)
    if (
        matrix.ndim != _MATRIX_DIMENSIONS
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
    ):
        message = "Spatial weights must be a nonempty square matrix."
        raise ValueError(message)
    dtype = jnp.result_type(matrix.dtype, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Spatial weights must be real-valued."
        raise ValueError(message)
    matrix = matrix.astype(dtype)
    n = matrix.shape[0]
    data, diagonal, row_sums, row_norm = _matrix_state(matrix)
    if not jnp.all(jnp.isfinite(data)):
        message = "Spatial weights must be finite."
        raise ValueError(message)
    if not allow_diagonal and jnp.any(diagonal != 0):
        message = "Nonzero diagonal requires allow_diagonal=True."
        raise ValueError(message)
    if not allow_isolates and jnp.any(jnp.asarray(row_norm) == 0):
        message = "Isolated rows require allow_isolates=True."
        raise ValueError(message)
    if normalize_rows:
        matrix, row_sums, row_norm = _normalize_rows(matrix, row_sums, row_norm)
    try:
        keys = jnp.arange(n) if unit_ids is None else jnp.asarray(unit_ids)
    except TypeError as error:
        message = (
            "unit_ids must be integer keys; encode labels with one stable codebook."
        )
        raise ValueError(message) from error
    if keys.shape != (n,) or not jnp.issubdtype(keys.dtype, jnp.integer):
        message = "unit_ids must be an aligned integer vector."
        raise ValueError(message)
    if jnp.unique(keys).size != n:
        message = "unit_ids must be unique."
        raise ValueError(message)
    return SpatialWeights(
        values=matrix,
        unit_ids=jnp.asarray(keys),
        row_sums=row_sums,
        limit=1 / jnp.maximum(jnp.max(row_norm), 1),
    )


def spatial_lag(weights: SpatialWeights, values: Array) -> Array:
    """Apply W to an aligned vector or matrix without normalizing the graph."""
    return weights.values @ values


def spatial_lags(weights: SpatialWeights, values: Array, *, order: int) -> Array:
    """Stack first through ``order`` lags; do not form sparse matrix powers."""
    if order < 1:
        message = "order must be positive."
        raise ValueError(message)

    def step(carry: Array, _: None) -> tuple[Array, Array]:
        lagged = spatial_lag(weights, carry)
        return lagged, lagged

    return jax.lax.scan(step, values, None, length=order)[1]


def spatial_logdet(weights: SpatialWeights, rho: Array) -> Array:
    """Return exact dense log|I-rho W|, with -inf at singularity.

    Sparse likelihoods need an explicitly chosen method; this entry point
    deliberately rejects sparse input rather than hiding an allocation.
    """
    if isinstance(weights.values, BCOO):
        message = "Exact dense likelihood requires explicitly prepared dense weights."
        raise TypeError(message)
    return jnp.linalg.slogdet(
        jnp.eye(weights.values.shape[0], dtype=weights.values.dtype)
        - rho * weights.values
    )[1]


def spatial_solve(
    weights: SpatialWeights,
    rho: Array,
    rhs: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 256,
) -> SolverResult[Array]:
    """Solve (I-rho W)x=rhs, retaining a checked residual and failure status."""
    return iterative_linear_solve(
        lambda x: x - rho * spatial_lag(weights, x),
        rhs,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )


def spatial_panel_lag(weights: Array | BCOO, values: Array) -> Array:
    """Apply fixed W or period-aligned W_t to time-major (T,N,...) data.

    Unit order must already agree for every period. Supports fixed-capacity
    batched BCOO values without materializing an NT-by-NT block operator.
    """
    if weights.ndim == _MATRIX_DIMENSIONS:

        def apply_fixed(x: Array) -> Array:
            return weights @ x

        return jax.vmap(apply_fixed)(values)
    if weights.ndim != _PANEL_DIMENSIONS or weights.shape[0] != values.shape[0]:
        message = "Use fixed (N,N) or time-aligned (T,N,N) weights."
        raise ValueError(message)

    def apply_period(w: Array | BCOO, x: Array) -> Array:
        return w @ x

    return jax.vmap(apply_period)(weights, values)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservedSpatialPeriod:
    """One actual graph and explicitly observed outcomes in its node order.

    Entity positions index a caller-declared global roster. A graph may omit
    entities that are inactive, or retain active nodes whose outcomes are
    missing. The distinction is represented by the graph and observed indices.
    """

    weights: SpatialWeights
    entity_indices: tuple[int, ...] = field(metadata={"static": True})
    observed_indices: tuple[int, ...] = field(metadata={"static": True})
