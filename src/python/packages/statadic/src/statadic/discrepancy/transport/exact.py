# jaxtyping shape strings are runtime contracts.
"""— exact balanced empirical Wasserstein-p via the Hungarian algorithm.

Consumes
--------
two equal-cardinality row samples, read as equal-weight empirical measures

Produces
--------
a scalar ``W_p`` and the square ground-cost matrix it is solved from

Order
-----
``W_p`` assigns on the ground cost raised to ``p``, averages the matched
powered costs, and takes the ``p``-th root. Different orders generally solve
different assignment problems, so the order cannot be applied after the fact
to a matching optimized for another order.

Ownership
---------
This member of :mod:`statadic.discrepancy.transport` forms the full ``n x n``
ground cost and solves the uniform assignment problem with
:func:`optax.assignment.hungarian_algorithm`, the same JAX implementation used
by :func:`ott.tools.unreg.hungarian`. The univariate and sliced methods have
different support and approximation contracts.

Transform boundary
------------------
The assignment core is fixed-shape and JAX-transformable. Eager entry points
validate inputs and delegate to the same core. Caller arrays and ordinary JAX
promotion determine precision; no entry point changes the precision context.
Finite arithmetic can obscure assignment differences, especially for extreme
orders or cost ranges. The result-bearing core reports detected underflow;
absence of that warning does not certify assignment accuracy.

Axioms — measured, and weaker than the name suggests
-----------------------------------------------------
``W_p`` for finite ``p >= 1`` over a metric ground cost is a metric on
``P_p``. The cloud convenience API imports Euclidean sphere-chord distance
from :mod:`statadic.geometry.ground.metrics` and requires **already unit-normalized rows**.
Normalization is a pipeline transformation that changes the estimand, so the
eager doors reject off-sphere rows instead of silently projecting them. The
exported array-level declaration remains a pseudometric because independent row
permutations represent the same empirical measure. On equal-cardinality
empirical measures on the unit sphere it is a genuine metric.

``PSEUDOMETRIC`` is off the linear marker chain, so the brand drops to
:class:`statadic.core.axioms.Premetric`; see the brand-down rule in
:mod:`statadic.core.axioms`.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Final

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Bool, Float  # noqa: TC002

from statadic.core.axioms import (
    Axioms,
    Premetric,
)
from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001
from statadic.geometry.ground.metrics import sphere_chord_distance
from statadic.sample.packed import PackedClouds  # runtime contract  # noqa: TC001

__all__ = [
    "EXACT_BALANCED_W1_AXIOMS",
    "EXACT_BALANCED_W1_BRAND",
    "EXACT_BALANCED_W2_AXIOMS",
    "EXACT_BALANCED_W2_BRAND",
    "EXACT_BALANCED_WASSERSTEIN_AXIOMS",
    "EXACT_BALANCED_WASSERSTEIN_BRAND",
    "EXPECTED_ARRAY_DIMENSIONS",
    "UNIT_NORM_ATOL",
    "WASSERSTEIN_ORDER_ONE",
    "WASSERSTEIN_ORDER_TWO",
    "EmptyOrNonMatrixSampleError",
    "ExactTransportResult",
    "InvalidCostMatrixError",
    "InvalidWassersteinOrderError",
    "NegativeCostError",
    "NonfiniteCostError",
    "NonfiniteSampleError",
    "OffUnitSphereSampleError",
    "UnbalancedSampleShapeError",
    "ZeroNormSampleError",
    "balanced_wasserstein1_from_cost",
    "balanced_wasserstein1_from_cost_jax",
    "balanced_wasserstein2_from_cost",
    "balanced_wasserstein2_from_cost_jax",
    "balanced_wasserstein_from_cost",
    "balanced_wasserstein_from_cost_jax",
    "balanced_wasserstein_from_cost_result",
    "chord_cost_matrix",
    "exact_balanced_wasserstein",
    "exact_balanced_wasserstein1",
    "exact_balanced_wasserstein1_jax",
    "exact_balanced_wasserstein2",
    "exact_balanced_wasserstein2_jax",
    "exact_balanced_wasserstein_jax",
]

EXPECTED_ARRAY_DIMENSIONS = 2
UNIT_NORM_ATOL: Final = 1e-6
WASSERSTEIN_ORDER_ONE: Final = 1.0
WASSERSTEIN_ORDER_TWO: Final = 2.0

#: Axioms :func:`exact_balanced_wasserstein1` satisfies on ``R^{n x d}`` row
#: samples of equal shape. Flags, never an alias name — see
#: :mod:`statadic.core.axioms`. ``IDENTITY`` is absent by measurement, not by
#: caution: row permutations identify the same empirical measure.
EXACT_BALANCED_W1_AXIOMS: Final = (
    Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY | Axioms.TRIANGLE
)

#: The strongest static marker :data:`EXACT_BALANCED_W1_AXIOMS` implies, for a
#: caller branding a :class:`statadic.core.matrices.DMat` assembled from this
#: dissimilarity. ``Semimetric`` would promise ``IDENTITY``, which does not hold.
EXACT_BALANCED_W1_BRAND: Final[type[Premetric]] = Premetric

#: Axioms :func:`exact_balanced_wasserstein2` satisfies, on the same row samples
#: and for the same measured reason as :data:`EXACT_BALANCED_W1_AXIOMS`. The
#: order enters the cost, not the axiom set: ``W2`` over a metric ground cost is
#: a metric on ``P_2``, and the Minkowski inequality carries the triangle
#: inequality through the quadratic aggregation.
EXACT_BALANCED_W2_AXIOMS: Final = (
    Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY | Axioms.TRIANGLE
)

#: The strongest static marker :data:`EXACT_BALANCED_W2_AXIOMS` implies. Same
#: brand-down reason as :data:`EXACT_BALANCED_W1_BRAND`.
EXACT_BALANCED_W2_BRAND: Final[type[Premetric]] = Premetric

#: Shared array-level law for every finite order ``p >= 1`` under the
#: sphere-chord ground metric. W1/W2 names remain compatibility projections.
EXACT_BALANCED_WASSERSTEIN_AXIOMS: Final = EXACT_BALANCED_W1_AXIOMS
EXACT_BALANCED_WASSERSTEIN_BRAND: Final[type[Premetric]] = Premetric


class EmptyOrNonMatrixSampleError(ValueError):
    """Raised when a sample is not a nonempty matrix."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("sample must be a nonempty two-dimensional array")


class NonfiniteSampleError(ValueError):
    """Raised when a sample contains a nonfinite coordinate."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("sample contains nonfinite values")


class ZeroNormSampleError(ValueError):
    """Raised when a sample contains a zero row."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("sample contains a zero-norm row")


class OffUnitSphereSampleError(ValueError):
    """Raised when a caller passes rows the pipeline did not normalize."""

    def __init__(self, worst_deviation: float) -> None:
        """Build the error with the measured unit-norm deviation."""
        super().__init__(
            "sample rows must already be L2-normalized by the calling pipeline; "
            f"worst |norm - 1| is {worst_deviation:.3e}"
        )


class UnbalancedSampleShapeError(ValueError):
    """Raised when balanced samples have different shapes."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__(
            "balanced Wasserstein requires equal-cardinality samples with equal "
            "feature width"
        )


class InvalidCostMatrixError(ValueError):
    """Raised when a balanced-Wasserstein cost matrix is structurally invalid."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("balanced Wasserstein requires a nonempty square cost matrix")


class NonfiniteCostError(ValueError):
    """Raised when a cost matrix contains a nonfinite entry."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("cost matrix contains nonfinite values")


class NegativeCostError(ValueError):
    """Raised when a ground-cost matrix contains a negative entry."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("cost matrix contains negative values")


class InvalidWassersteinOrderError(ValueError):
    """Raised when an eager caller requests an order outside ``[1, inf)``."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("Wasserstein order p must be finite and at least 1")


@static_jaxtyping("m d")
def _checked_unit_rows(sample: ArrayLike) -> Float[Array, "m d"]:
    """Validate and return a sample whose rows are already unit norm.

    Args:
        sample: A nonempty ``(n, d)`` array of observations.

    Returns:
        The same rows in the active caller dtype, unchanged.

    Raises:
        EmptyOrNonMatrixSampleError: If ``sample`` is not a nonempty matrix.
        NonfiniteSampleError: If any coordinate is not finite.
        ZeroNormSampleError: If any row has zero norm.
        OffUnitSphereSampleError: If a row is not unit norm within tolerance.

    """
    rows: Float[Array, "m d"] = jnp.asarray(sample)
    if (
        rows.ndim != EXPECTED_ARRAY_DIMENSIONS
        or rows.shape[0] == 0
        or rows.shape[1] == 0
    ):
        raise EmptyOrNonMatrixSampleError
    if not bool(jnp.all(jnp.isfinite(rows))):
        raise NonfiniteSampleError
    norms = jnp.linalg.norm(rows, axis=1)
    if bool(jnp.any(norms <= 0.0)):
        raise ZeroNormSampleError
    deviations = jnp.abs(norms - 1.0)
    if bool(jnp.any(deviations > UNIT_NORM_ATOL)):
        raise OffUnitSphereSampleError(float(deviations.max()))
    return rows


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExactTransportResult:
    """Distance and numerical evidence for one uniform assignment.

    ``input_valid`` concerns finite nonnegative costs and a finite order at
    least one. ``finite`` records whether the returned distance is finite.
    ``power_underflow`` detects positive entries lost while normalizing or
    powering either the assignment matrix or selected costs. It is a warning,
    not a test of whether the chosen assignment is correct: finite precision
    can hide distinctions without producing zero, and some platforms flush
    subnormal inputs before they can be detected.
    """

    value: Float[Array, "*batch"]
    input_valid: Bool[Array, "*batch"]
    finite: Bool[Array, "*batch"]
    power_underflow: Bool[Array, "*batch"]


def _scaled_power(cost: Array, order: Array) -> tuple[Array, Array, Array]:
    """Scale nonnegative costs before powering and report lost positives."""
    scale = jnp.max(cost)
    safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    normalized = cost / safe_scale
    powered = jnp.where(
        order == WASSERSTEIN_ORDER_ONE,
        normalized,
        jnp.where(
            order == WASSERSTEIN_ORDER_TWO,
            jnp.square(normalized),
            jnp.power(normalized, order),
        ),
    )
    return scale, powered, jnp.any((cost > 0) & (powered == 0))


def balanced_wasserstein_from_cost_result(
    cost: Float[Array, "..."],
    p: float | Float[Array, "..."],
) -> ExactTransportResult:
    """Solve uniform balanced ``W_p`` and retain numerical evidence.

    Nonempty square cost and scalar order shapes are checked at tracing time;
    their numeric domains are recorded in ``input_valid``. Invalid numeric
    inputs return NaN after a safe computation; eager validation raises before
    reaching here.
    A common scale preserves the powered assignment objective in exact
    arithmetic. A second scale on the selected original costs stabilizes their
    rooted mean when unmatched costs are much larger. Underresolution can
    still change an assignment, so inspect the evidence and qualification.
    """
    if (
        cost.ndim != EXPECTED_ARRAY_DIMENSIONS
        or cost.shape[0] == 0
        or (cost.shape[0] != cost.shape[1])
    ):
        raise InvalidCostMatrixError
    order = jnp.asarray(p)
    if order.ndim != 0:
        raise InvalidWassersteinOrderError
    valid_cost = jnp.isfinite(cost) & (cost >= 0)
    valid_order = jnp.isfinite(order) & (order >= 1)
    input_valid = jnp.all(valid_cost) & valid_order
    safe_cost = jnp.where(valid_cost, cost, jnp.zeros_like(cost))
    order = jnp.where(valid_order, order, jnp.ones_like(order))
    _, powered, assignment_underflow = _scaled_power(safe_cost, order)
    assignment = optax.assignment.hungarian_algorithm(powered)
    matched_scale, matched_power, matched_underflow = _scaled_power(
        safe_cost[assignment], order
    )
    matched_mean = jnp.mean(matched_power)
    rooted = jnp.where(
        order == WASSERSTEIN_ORDER_ONE,
        matched_mean,
        jnp.where(
            order == WASSERSTEIN_ORDER_TWO,
            jnp.sqrt(matched_mean),
            jnp.power(matched_mean, 1.0 / order),
        ),
    )
    value = jnp.where(input_valid, matched_scale * rooted, jnp.nan)
    return ExactTransportResult(
        value=value,
        input_valid=input_valid,
        finite=jnp.isfinite(value),
        power_underflow=assignment_underflow | matched_underflow,
    )


@static_jaxtyping("n")
def balanced_wasserstein_from_cost_jax(
    cost: Float[Array, "n n"],
    p: float | Float[Array, ""],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Project the distance from :func:`balanced_wasserstein_from_cost_result`."""
    return balanced_wasserstein_from_cost_result(cost, p).value


@static_jaxtyping("n")
def balanced_wasserstein1_from_cost_jax(
    cost: Float[Array, "n n"],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Compatibility specialization of uniform balanced ``W_p`` at ``p=1``."""
    return balanced_wasserstein_from_cost_jax(cost, WASSERSTEIN_ORDER_ONE)


@static_jaxtyping("n d")
def exact_balanced_wasserstein_jax(  # noqa: PLR0914
    left: "PackedClouds",
    right: "PackedClouds",
    p: float | Float[Array, ""],
) -> Array:
    """Compute exact ``W_p`` under sphere-chord cost over broadcast cloud axes."""
    if (
        left.data.ndim != EXPECTED_ARRAY_DIMENSIONS
        or right.data.ndim != EXPECTED_ARRAY_DIMENSIONS
    ):
        return left.map_aligned(
            right, lambda a, b: exact_balanced_wasserstein_jax(a, b, p)
        )
    capacity = max(left.data.shape[-2], right.data.shape[-2])
    left_coordinates = jnp.pad(
        left.coordinates, ((0, capacity - left.data.shape[-2]), (0, 0))
    )
    right_coordinates = jnp.pad(
        right.coordinates, ((0, capacity - right.data.shape[-2]), (0, 0))
    )
    costs = sphere_chord_distance(left_coordinates, right_coordinates)
    left_active = jnp.arange(capacity) < left.counts
    right_active = jnp.arange(capacity) < right.counts
    active = left_active[:, None] & right_active[None, :]
    mixed = left_active[:, None] ^ right_active[None, :]
    order = jnp.asarray(p)
    if order.ndim != 0:
        raise InvalidWassersteinOrderError
    valid_order = jnp.isfinite(order) & (order >= 1)
    safe_order = jnp.where(valid_order, order, jnp.ones_like(order))
    active_costs = jnp.where(active, costs, 0)
    finite_costs = jnp.all(jnp.isfinite(active_costs))
    safe_costs = jnp.where(jnp.isfinite(active_costs), active_costs, 0)
    _, powered, _ = _scaled_power(safe_costs, safe_order)
    assignment = optax.assignment.hungarian_algorithm(
        jnp.where(mixed, capacity + 1.0, powered)
    )
    selected = jnp.where(left_active[assignment[0]], safe_costs[assignment], 0)
    scale, selected_power, _ = _scaled_power(selected, safe_order)
    mean = jnp.sum(selected_power) / jnp.maximum(left.counts, 1)
    rooted = jnp.where(
        safe_order == WASSERSTEIN_ORDER_ONE,
        mean,
        jnp.where(
            safe_order == WASSERSTEIN_ORDER_TWO,
            jnp.sqrt(mean),
            jnp.power(mean, 1.0 / safe_order),
        ),
    )
    value = scale * rooted
    valid = (
        left.valid
        & right.valid
        & (left.counts == right.counts)
        & valid_order
        & finite_costs
    )
    valid &= left.is_uniform & right.is_uniform
    return jnp.where(valid, value, jnp.nan)


@static_jaxtyping("n d")
def exact_balanced_wasserstein1_jax(
    left: "PackedClouds",
    right: "PackedClouds",
) -> Array:
    """Compute exact W1 over unit-sphere supports and broadcast cloud axes."""
    return exact_balanced_wasserstein_jax(left, right, WASSERSTEIN_ORDER_ONE)


@static_jaxtyping("n")
def balanced_wasserstein2_from_cost_jax(
    cost: Float[Array, "n n"],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Solve uniform balanced W2 with the exact transformable Hungarian core.

    The assignment runs on the squared ground cost, which is the objective W2
    minimizes; squaring after a W1 assignment would report a feasible but
    suboptimal transport plan.
    """
    return balanced_wasserstein_from_cost_jax(cost, WASSERSTEIN_ORDER_TWO)


@static_jaxtyping("n d")
def exact_balanced_wasserstein2_jax(
    left: "PackedClouds",
    right: "PackedClouds",
) -> Array:
    """Compute exact W2 over unit-sphere supports and broadcast cloud axes."""
    return exact_balanced_wasserstein_jax(left, right, WASSERSTEIN_ORDER_TWO)


@static_jaxtyping("m n d")
def chord_cost_matrix(left: ArrayLike, right: ArrayLike) -> Float[Array, "m n"]:
    """Return the chord-cost matrix in the caller's configured JAX precision.

    The samples must have the same positive cardinality and feature width.
    Chord distance between unit rows is
    ``sqrt(2 - 2 * dot(left_i, right_j))``.

    Args:
        left: An ``(n, d)`` sample.
        right: An ``(n, d)`` sample with the same shape as ``left``.

    Returns:
        The ``(n, n)`` ground-cost matrix of chord distances between the
        already-unit rows of ``left`` and of ``right``.

    Raises:
        UnbalancedSampleShapeError: If the two samples do not share a shape.

    """
    left_shape = jnp.shape(left)
    right_shape = jnp.shape(right)
    if (
        len(left_shape) == EXPECTED_ARRAY_DIMENSIONS
        and len(right_shape) == EXPECTED_ARRAY_DIMENSIONS
        and left_shape != right_shape
    ):
        raise UnbalancedSampleShapeError
    left_unit: Float[Array, "m d"] = _checked_unit_rows(left)
    right_unit: Float[Array, "n d"] = _checked_unit_rows(right)
    return sphere_chord_distance(left_unit, right_unit)


def _checked_order(p: object) -> float:
    """Return a finite Wasserstein order on the metric range ``[1, inf)``."""
    if isinstance(p, bool) or not isinstance(p, Real):
        raise InvalidWassersteinOrderError
    order = float(p)
    if not jnp.isfinite(order) or order < 1.0:
        raise InvalidWassersteinOrderError
    return order


@static_jaxtyping("n")
def _checked_cost_matrix(cost: ArrayLike) -> Float[Array, "n n"]:
    """Validate and return a floating square ground-cost matrix.

    Integral inputs are promoted before entering the JAX computation.
    Caller-provided floating dtypes are retained; the shared core scales finite
    costs before applying its power, avoiding avoidable active-dtype overflow.

    Args:
        cost: A nonempty square ``(n, n)`` matrix of nonnegative ground costs.

    Returns:
        The validated matrix in its caller-provided floating dtype; integral
        inputs follow JAX's ordinary floating promotion.

    Raises:
        InvalidCostMatrixError: If ``cost`` is empty or not square.
        NonfiniteCostError: If an entry is nonfinite.
        NegativeCostError: If any entry is negative.

    """
    raw = jnp.asarray(cost)
    if (
        raw.ndim != EXPECTED_ARRAY_DIMENSIONS
        or raw.shape[0] == 0
        or raw.shape[0] != raw.shape[1]
    ):
        raise InvalidCostMatrixError
    matrix: Float[Array, "n n"] = jnp.asarray(raw) * 1.0
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise NonfiniteCostError
    if bool(jnp.any(matrix < 0.0)):
        raise NegativeCostError
    return matrix


def balanced_wasserstein_from_cost(cost: ArrayLike, p: object) -> float:
    """Solve exact uniform balanced ``W_p`` from a ground-cost matrix.

    Args:
        cost: A nonempty square matrix of nonnegative ground costs.
        p: A finite Wasserstein order at least one.

    Returns:
        The rooted optimal mean ``p``-power cost between two uniform empirical
        measures of equal cardinality.

    """
    order = _checked_order(p)
    matrix = _checked_cost_matrix(cost)
    return float(balanced_wasserstein_from_cost_jax(matrix, order))


def balanced_wasserstein1_from_cost(cost: ArrayLike) -> float:
    """Solve uniform balanced W1 exactly from a square ground-cost matrix.

    Args:
        cost: A nonempty square ``(n, n)`` matrix of nonnegative ground costs.

    Returns:
        The mean cost of the optimal assignment — the exact ``W1`` between two
        uniform empirical measures of ``n`` atoms each.

    """
    return balanced_wasserstein_from_cost(cost, WASSERSTEIN_ORDER_ONE)


def balanced_wasserstein2_from_cost(cost: ArrayLike) -> float:
    """Solve uniform balanced W2 exactly from a square ground-cost matrix.

    Args:
        cost: A nonempty square ``(n, n)`` matrix of nonnegative ground costs.

    Returns:
        The root-mean squared cost of the assignment optimal for the squared
        cost — the exact ``W2`` between two uniform empirical measures of ``n``
        atoms each.

    """
    return balanced_wasserstein_from_cost(cost, WASSERSTEIN_ORDER_TWO)


def exact_balanced_wasserstein(
    left: "PackedClouds",
    right: "PackedClouds",
    p: object,
) -> float:
    """Exact uniform empirical ``W_p`` under sphere-chord ground cost.

    The samples must have equal cardinality and width, contain finite nonzero
    rows, and already lie on the L2 unit sphere. The eager boundary validates
    those conditions and ``p >= 1``.
    """
    order = _checked_order(p)
    if (
        left.data.ndim != EXPECTED_ARRAY_DIMENSIONS
        or right.data.ndim != EXPECTED_ARRAY_DIMENSIONS
        or not bool(left.valid & right.valid)
        or int(left.counts) != int(right.counts)
    ):
        msg = "Exact assignment requires two valid equal-cardinality distributions"
        raise ValueError(msg)
    n = int(left.counts)
    if not bool(left.is_uniform & right.is_uniform):
        msg = "Exact assignment requires uniform atom masses"
        raise ValueError(msg)
    return balanced_wasserstein_from_cost(
        chord_cost_matrix(left.coordinates[:n], right.coordinates[:n]), order
    )


def exact_balanced_wasserstein1(left: "PackedClouds", right: "PackedClouds") -> float:
    """Exact empirical balanced W1 under uniform weights and chord ground cost.

    Satisfies :data:`EXACT_BALANCED_W1_AXIOMS` — ``NONNEGATIVE | SYMMETRY |
    TRIANGLE``. Rows must already be normalized by the caller. It is a metric
    on equal-cardinality empirical measures on the unit sphere.

    Args:
        left: An ``(n, d)`` sample, read as ``n`` equally-weighted atoms.
        right: An ``(n, d)`` sample with the same shape as ``left``.

    Returns:
        The exact ``W1`` between the two empirical measures, invariant to the
        row order of either argument.

    """
    return exact_balanced_wasserstein(left, right, WASSERSTEIN_ORDER_ONE)


def exact_balanced_wasserstein2(left: "PackedClouds", right: "PackedClouds") -> float:
    """Exact empirical balanced W2 under uniform weights and chord ground cost.

    Satisfies :data:`EXACT_BALANCED_W2_AXIOMS` — ``NONNEGATIVE | SYMMETRY |
    TRIANGLE``. Rows must already be normalized by the caller. It is a metric
    on equal-cardinality empirical measures on the unit sphere, and dominates
    :func:`exact_balanced_wasserstein1` by Cauchy--Schwarz.

    Args:
        left: An ``(n, d)`` sample, read as ``n`` equally-weighted atoms.
        right: An ``(n, d)`` sample with the same shape as ``left``.

    Returns:
        The exact ``W2`` between the two empirical measures, invariant to the
        row order of either argument.

    """
    return exact_balanced_wasserstein(left, right, WASSERSTEIN_ORDER_TWO)
