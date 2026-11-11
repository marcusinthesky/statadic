"""Rank-clean regression-design operators for undirected dyadic panels.

These functions construct or residualize a design matrix.  They deliberately
know nothing about a fitted model, bootstrap schedule, paper-specific regressor
names, or artifact format; those concerns begin at :mod:`statadic.model.dyadic`.
"""

from __future__ import annotations

import operator
from collections.abc import (  # noqa: TC003  # runtime Protocol
    Hashable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # noqa: TC002  # runtime jaxtyping contract

__all__ = [
    "ArrayView",
    "FWLResiduals",
    "LabelArray",
    "NodeIndex",
    "drop_collinear_columns",
    "fwl_coefficient",
    "fwl_residualize",
    "node_labels_from_array",
    "symmetric_dyadic_effects",
]

_MATRIX_DIMENSIONS = 2
_MIN_ENDPOINTS = 2


@runtime_checkable
class ArrayView(Protocol):
    """Structural view of an aligned numeric dyadic column (NumPy or JAX).

    Deliberately narrower than ``jax.typing.ArrayLike``, which admits bare
    scalars and so erases the shape and indexing guarantees every reader
    depends on. Deliberately wider than ``jax.Array``, because external adapters
    supply NumPy panels. Host node labels use ``LabelArray`` and explicit decoding
    instead. Only the container surface is constrained here, never the dtype.
    """

    # `Any` is load-bearing on the element-typed members: NumPy and JAX
    # disagree on their index and element types. Constraining them here would
    # reject one backend or the other.

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the column's dimensions."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        ...

    def __getitem__(self, key: Any, /) -> Any:  # noqa: ANN401
        """Return one element or slice."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the leading axis."""
        ...

    def __len__(self) -> int:
        """Return the length of the leading axis."""
        ...

    def tolist(self) -> Any:  # noqa: ANN401
        """Return the contents as nested host lists."""
        ...


@runtime_checkable
class LabelArray(Protocol):
    """An external array whose erased host labels require explicit decoding."""

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions."""
        ...

    def tolist(self) -> object:
        """Return host values whose shape and label types remain unchecked."""
        ...


def node_labels_from_array(value: LabelArray) -> tuple[Hashable, ...]:
    """Decode a one-dimensional external array into validated host node labels."""
    if value.ndim != 1:
        message = "dyadic endpoint IDs must be a one-dimensional array"
        raise ValueError(message)
    values = value.tolist()
    if not isinstance(values, list):
        message = "dyadic endpoint arrays must decode to a flat list"
        # External label failures share the NodeIndex value-validation contract.
        raise ValueError(message)  # noqa: TRY004
    labels: list[Hashable] = []
    for node in values:
        if not isinstance(node, Hashable):
            message = "dyadic endpoint IDs must be non-missing hashable scalar values"
            # External label failures share the NodeIndex value-validation contract.
            raise ValueError(message)  # noqa: TRY004
        labels.append(_validate_node_id(node))
    return tuple(labels)


def _node_sort_key(node: Hashable) -> tuple[str, str]:
    """Order unlike ID types; cross-process stability follows their ``repr``."""
    node_type = type(node)
    return f"{node_type.__module__}.{node_type.__qualname__}", repr(node)


def _validate_node_id[NodeIdT: Hashable](node: NodeIdT) -> NodeIdT:
    """Return one usable entity ID or reject missing/non-hashable values."""
    if node is None:
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message)
    try:
        hash(node)
    except (TypeError, ValueError) as error:
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message) from error
    try:
        reflexive = operator.eq(node, node)
    except (TypeError, ValueError) as error:
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message) from error
    if not isinstance(reflexive, bool) or not bool(reflexive):
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message)
    return node


@dataclass(frozen=True)
class NodeIndex[NodeIdT: Hashable = Hashable]:
    """Canonical entity IDs and their shared dyadic-design column mapping.

    Canonical ordering uses an ID's qualified type and representation, rather
    than requiring IDs of unrelated types to implement a common ``<`` order.
    It is cross-process deterministic for primitive IDs and custom IDs whose
    qualified type name and ``repr`` are themselves stable; arbitrary custom
    representations can only promise process-local ordering. The representation
    key must distinguish unequal IDs, so ambiguous keys fail closed instead of
    making endpoint orientation choose the omitted effect column.
    """

    node_ids: tuple[NodeIdT, ...]

    def __post_init__(self) -> None:
        """Validate, deduplicate, and canonically order entity identifiers."""
        validated = tuple(_validate_node_id(node) for node in self.node_ids)
        try:
            unique = tuple(dict.fromkeys(validated))
        except (TypeError, ValueError) as error:
            message = "dyadic endpoint IDs must have scalar equality semantics"
            raise ValueError(message) from error
        if len(unique) != len(validated):
            message = "node index IDs must be unique"
            raise ValueError(message)
        if len(unique) < _MIN_ENDPOINTS:
            message = "dyadic data require at least two distinct endpoint IDs"
            raise ValueError(message)
        keyed = [(_node_sort_key(node), node) for node in unique]
        if len({key for key, _node in keyed}) != len(keyed):
            message = (
                "unequal endpoint IDs must have distinct canonical representations"
            )
            raise ValueError(message)
        canonical = tuple(
            node for _key, node in sorted(keyed, key=lambda item: item[0])
        )
        object.__setattr__(self, "node_ids", canonical)

    def __len__(self) -> int:
        """Return the number of indexed entities."""
        return len(self.node_ids)

    @staticmethod
    def from_endpoints[IdT: Hashable](
        endpoint_i: Sequence[IdT],
        endpoint_j: Sequence[IdT],
    ) -> NodeIndex[IdT]:
        """Build the canonical index for one aligned distinct-node dyadic panel."""
        if len(endpoint_i) != len(endpoint_j):
            message = "dyadic endpoint arrays must be aligned one-dimensional vectors"
            raise ValueError(message)
        values = [*endpoint_i, *endpoint_j]
        validated = [_validate_node_id(value) for value in values]
        try:
            unique = tuple(dict.fromkeys(validated))
        except (TypeError, ValueError) as error:
            message = "dyadic endpoint IDs must have scalar equality semantics"
            raise ValueError(message) from error
        index = NodeIndex(unique)
        i_position = index.positions_for(endpoint_i)
        j_position = index.positions_for(endpoint_j)
        if bool(jnp.any(i_position == j_position)):
            message = "undirected dyadic samples cannot contain self-dyads"
            raise ValueError(message)
        return index

    @staticmethod
    def from_arrays(
        endpoint_i: LabelArray, endpoint_j: LabelArray
    ) -> NodeIndex[Hashable]:
        """Decode external label arrays without inventing their erased ID type."""
        return NodeIndex.from_endpoints(
            node_labels_from_array(endpoint_i), node_labels_from_array(endpoint_j)
        )

    def positions_for(self, endpoint_ids: Sequence[NodeIdT]) -> jax.Array:
        """Return integer positions for IDs under this exact canonical mapping."""
        lookup = {node: position for position, node in enumerate(self.node_ids)}
        positions: list[int] = []
        for raw_node in endpoint_ids:
            node = _validate_node_id(raw_node)
            try:
                positions.append(lookup[node])
            except (KeyError, TypeError, ValueError) as error:
                message = f"endpoint ID is absent from the node index: {node!r}"
                raise ValueError(message) from error
        return jnp.asarray(positions, dtype=jnp.int64)


@jax.jit
def _rank_profile_gram_schmidt(xf: jax.Array, threshold: jax.Array) -> jax.Array:
    """Return the declared-order rank profile in one traced pass.

    Ask whether each column leaves the span of its accepted predecessors,
    carrying an
    orthonormal basis instead of re-factorizing a growing candidate.  Rejected
    slots of ``basis`` stay exactly zero, so ``basis @ (basis.T @ column)``
    projects onto the accepted prefix and nothing else; no masking is needed.
    The reprojection is Kahan's "twice is enough": one repeat restores
    orthogonality to machine precision after cancellation, which plain
    Gram-Schmidt loses.

    ``lax.scan`` keeps every shape fixed, so this is one compilation and, unlike
    the ``matrix_rank`` loop, carries no per-column device synchronization --
    the whole profile is traceable.  ``threshold`` is a traced argument rather
    than a closed-over Python float on purpose: as a constant it lands in the
    jaxpr and re-keys the cache on every call, which cost one recompilation per
    invocation (measured 20 compilations across 20 calls) and gave back most of
    what the single pass buys.

    The accept test is *not* the same predicate as growing ``matrix_rank``.
    Greedy asks whether ``sigma_{k+1}`` clears the threshold; this asks whether
    the orthogonal residual norm does, and ``sigma_min([A a]) <= dist(a,
    span(A))``, so this is the more permissive of the two.  They can only
    disagree when a singular value lands near the threshold, which is a regime
    the callers do not reach -- see :func:`drop_collinear_columns`.

    Rank decisions use the supplied design precision.
    """
    rows, width = xf.shape
    capacity = min(rows, width)
    if capacity == 0:
        # ``scan`` still traces its body once at zero length, and indexing an
        # empty basis raises rather than being skipped.
        return jnp.zeros((width,), dtype=bool)

    def step(
        carry: tuple[jax.Array, jax.Array], column: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        basis, filled = carry
        residual = column - basis @ (basis.T @ column)
        residual = residual - basis @ (basis.T @ residual)
        norm = jnp.linalg.norm(residual)
        # ``filled < capacity`` cannot bind on a real design -- no more than
        # ``capacity`` columns can be independent -- but without it a numerical
        # accept past the last slot would clamp onto, and destroy, an existing
        # basis vector rather than being ignored.
        accept = (norm > threshold) & (filled < capacity)
        unit = jnp.where(accept, residual / jnp.where(accept, norm, 1.0), 0.0)
        index = jnp.minimum(filled, capacity - 1)
        return (
            jnp.where(accept, basis.at[:, index].set(unit), basis),
            filled + accept.astype(filled.dtype),
        ), accept

    initial = (jnp.zeros((rows, capacity), dtype=xf.dtype), jnp.asarray(0))
    _carry, accepted = jax.lax.scan(step, initial, xf.T)
    return accepted


def drop_collinear_columns(x: ArrayLike, tol: float | None = None) -> jax.Array:
    """Return an ordered maximal linearly independent set of column indices.

    Greedy rank growth keeps the first representative of each collinear column
    family, so the declared column order is preserved and a named regressor is
    never dropped in favour of a generated effect that follows it.

    :func:`_rank_profile_gram_schmidt` answers that in one traced pass;
    the ``matrix_rank`` reference in ``tests.operators.test_design`` is used
    for the parity sweep. The two use different
    numerical predicates and agree on every structural, random, and wide case in
    that sweep; they diverge only when a singular value sits within about two
    orders of magnitude of ``tol * ||x||_2``.  Across 600 real node-bootstrap
    designs -- both the weighted full design and the effect-only design of
    ``model._dyadic.inference`` -- no singular value landed in that band: the
    closest retained one was 2.6e8 times the threshold and the closest dropped
    nonzero one 3.6e-6 times it, because collinearity here is structural (a
    zero-weight node zeroes a column exactly) rather than marginal.  Condition
    number is *not* the diagnostic to check this with -- every one of those
    designs exceeded 1e9, and the worst reached 1e99, purely from exact zeros.

    Neither QR variant can replace the loop, and the tests pin both
    counterexamples:

    * **Unpivoted**, reading the profile off ``|diag(R)|``, under-selects.  A
      dependent column consumes its diagonal row whether or not it adds rank,
      which shifts every later independent column off the diagonal.  On
      ``[[1, 1, 0], [0, 0, 1]]`` the duplicate at index 1 burns row 1 and the
      diagonal reads ``[1, 0]``, yielding ``[0]`` where the rank-2 contract
      wants ``[0, 2]``.
    * **Column-pivoted** (``geqp3``, ``jax.lax.linalg.qr(..., pivoting=True)``)
      reorders by norm, so it returns *a* maximal independent set rather than
      *the* declared-order one.  On ``[[1, 0, 1], [0, 1, 1]]`` it prefers the
      largest-norm column and yields ``[0, 2]`` where the contract wants
      ``[0, 1]``.

    The 1e9 conditioning above is not a curiosity: under float32 the threshold
    test has no signal left, and
    a full
    ``dvc repro`` with the caller flag off rejected every one of 1999
    node-bootstrap draws in ``p1_dyadic_confound``.  An array born under x64 also
    keeps its dtype *label* outside the scope while later ops silently compute
    in float32 (measured 2.5e-8 relative error on the SSE path), so a dtype
    assertion alone does not detect the loss.

    Args:
        x: Two-dimensional design matrix.
        tol: Relative matrix-rank tolerance, scaled by ``max(||x||_2, 1)``.
            By default, use ``max(1e-10, max(x.shape) * eps(dtype))``; an
            explicit tolerance is used unchanged.

    Returns:
        Integer indices of retained columns, in their original order.

    """
    xa = jnp.asarray(x)
    if xa.ndim != _MATRIX_DIMENSIONS:
        message = f"design must be two-dimensional; received shape {xa.shape}"
        raise ValueError(message)
    xf = jnp.asarray(x) * 1.0
    scale = max(float(jnp.linalg.norm(xf, ord=2)), 1.0)
    relative_tolerance = (
        max(1e-10, max(xf.shape) * float(jnp.finfo(xf.dtype).eps))
        if tol is None
        else tol
    )
    threshold = jnp.asarray(relative_tolerance * scale, dtype=xf.dtype)
    accepted = _rank_profile_gram_schmidt(xf, threshold)
    return jnp.flatnonzero(accepted).astype(jnp.int64)


def symmetric_dyadic_effects[NodeIdT: Hashable](
    endpoint_i: Sequence[NodeIdT],
    endpoint_j: Sequence[NodeIdT],
    *,
    name_prefix: str = "node_effect",
    node_index: NodeIndex[NodeIdT] | None = None,
) -> tuple[jax.Array, list[str]]:
    """Construct additive undirected endpoint effects with one base omitted.

    Row ``r`` contains ``1{endpoint_i[r] = k} + 1{endpoint_j[r] = k}`` for
    every endpoint ``k`` except the first endpoint in the canonical node index.

    Args:
        endpoint_i: First endpoint label for each dyad.
        endpoint_j: Second endpoint label for each dyad.
        name_prefix: Prefix used for the non-base endpoint-effect labels.
        node_index: Optional validated mapping shared with fit/bootstrap code.

    Returns:
        Effect matrix and aligned ``<name_prefix>[...]`` column labels.

    Raises:
        ValueError: If endpoint arrays are not aligned or name fewer than two
            distinct endpoints.

    """
    if not isinstance(name_prefix, str) or not name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise ValueError(message)
    index = node_index or NodeIndex.from_endpoints(endpoint_i, endpoint_j)
    i_position = index.positions_for(endpoint_i)
    j_position = index.positions_for(endpoint_j)
    observed_positions = set(i_position.tolist()) | set(j_position.tolist())
    if observed_positions != set(range(len(index))):
        message = "node index must exactly match the observed dyadic endpoint IDs"
        raise ValueError(message)
    if bool(jnp.any(i_position == j_position)):
        message = "undirected dyadic samples cannot contain self-dyads"
        raise ValueError(message)
    columns = [
        (i_position == position).astype(jnp.int32)
        + (j_position == position).astype(jnp.int32)
        for position in range(1, len(index))
    ]
    effects = jnp.column_stack(columns)
    return effects, [f"{name_prefix}[{endpoint}]" for endpoint in index.node_ids[1:]]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FWLResiduals:
    """Both nuisance-projected series and their residual-on-residual slope.

    All fields are JAX-array pytree leaves. Residuals have the input row shape;
    ``slope`` is scalar, or has the batch shape after ``vmap``.
    """

    outcome_residual: jax.Array
    focal_residual: jax.Array
    slope: jax.Array


def fwl_residualize(x: ArrayLike, y: ArrayLike, focal_column: int) -> FWLResiduals:
    """Project outcome and focal regressor off the same nuisance design.

    A single least-squares solve projects both series using ``rcond=None``.
    This operation accepts fixed-shape ``jit`` and leading-axis ``vmap`` calls
    with a static ``focal_column``. No derivative contract across rank changes
    is implied. Float64 computation requires caller-owned
    ``jax_enable_x64=True``; this function never changes that configuration.

    Args:
        x: Full-rank or already rank-cleaned design matrix, including any
            intercept and nuisance effects supplied by the caller.
        y: Aligned outcome vector.
        focal_column: Column of ``x`` to residualize against all other columns.

    Returns:
        Both residual series and their slope. The slope is ``NaN`` when the
        residual sum of squares is zero. This is an exact zero guard, not a
        numerical rank or identification diagnostic.

    """
    xa = jnp.asarray(x)
    ya = jnp.asarray(y)
    if xa.ndim != _MATRIX_DIMENSIONS or ya.ndim != 1 or xa.shape[0] != ya.shape[0]:
        message = "design and outcome must be an aligned matrix/vector pair"
        raise ValueError(message)
    xf = jnp.asarray(x)
    yf = jnp.asarray(y)
    nuisance = jnp.delete(xf, focal_column, axis=1)
    focal = xf[:, focal_column]
    responses = jnp.column_stack([yf, focal])
    residuals = (
        responses - nuisance @ jnp.linalg.lstsq(nuisance, responses, rcond=None)[0]
    )
    y_residual, focal_residual = residuals[:, 0], residuals[:, 1]
    denominator = focal_residual @ focal_residual
    slope = jnp.where(
        denominator <= 0.0,
        jnp.nan,
        (focal_residual @ y_residual) / denominator,
    )
    return FWLResiduals(y_residual, focal_residual, slope)


def fwl_coefficient(x: ArrayLike, y: ArrayLike, focal_column: int) -> float:
    """Return the shared FWL slope as an eager Python float.

    See :func:`fwl_residualize` for the design, precision and zero-variation
    contract. Use its array-valued result inside ``jit`` or ``vmap``.
    """
    return float(fwl_residualize(x, y, focal_column).slope)
