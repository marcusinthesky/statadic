"""Balanced transportation linear programs with a device-native simplex method.

The supplied rectangular cost matrix is the objective itself: this numerical
solver neither powers nor roots it. Both marginals are probability vectors.
The north-west corner initializes a spanning-tree basis, including zero-flow
basic edges. Bland's entering and leaving rules resolve degenerate pivots.

Fixed array shapes support JIT and vmap. Discrete pivots are not a qualified
derivative of an optimal transport solution. A finite iteration budget can
return an unavailable objective; the retained plan and residuals explain why.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from statadic.core.typing import Array  # noqa: TC001

__all__ = ["TransportationResult", "solve_balanced_transport"]

_MATRIX_DIMENSIONS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TransportationResult:
    """One primal plan, a feasible dual witness, and convergence evidence.

    ``value`` is NaN unless the input probability laws and numerical solve are
    valid. ``gap`` is the primal cost minus the retained dual lower bound, on
    the original cost scale. ``residual`` is the largest marginal discrepancy.
    A tied optimum can have several valid plans; this is one deterministic
    witness. Neither a finite plan nor a small marginal residual alone proves
    objective optimality.
    """

    value: Array
    plan: Array
    row_potentials: Array
    column_potentials: Array
    gap: Array
    residual: Array
    converged: Array
    iterations: Array
    input_valid: Array
    finite: Array


def _north_west_basis(a: Array, b: Array) -> tuple[Array, Array]:
    """Initialize a feasible spanning tree without dropping degenerate edges."""
    rows, columns = a.size, b.size
    plan = jnp.zeros((rows, columns), dtype=jnp.result_type(a, b))
    basis = jnp.zeros((rows, columns), dtype=jnp.bool_)

    def allocate(
        _: Array, state: tuple[Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        plan, basis, supply, demand, row, column = state
        mass = jnp.minimum(supply[row], demand[column])
        plan = plan.at[row, column].set(mass)
        basis = basis.at[row, column].set(True)
        supply = supply.at[row].add(-mass)
        demand = demand.at[column].add(-mass)
        move_row = (column == columns - 1) | (
            (supply[row] <= demand[column]) & (row < rows - 1)
        )
        row = jnp.minimum(row + move_row, rows - 1)
        column = jnp.minimum(column + ~move_row, columns - 1)
        return plan, basis, supply, demand, row, column

    plan, basis, *_ = jax.lax.fori_loop(
        0,
        rows + columns - 1,
        allocate,
        (plan, basis, a, b, jnp.asarray(0), jnp.asarray(0)),
    )
    return plan, basis


def _tree_potentials(cost: Array, basis: Array) -> tuple[Array, Array]:
    """Solve u_i + v_j = c_ij over a spanning-tree basis."""
    rows, columns = cost.shape
    u, v = jnp.zeros(rows, cost.dtype), jnp.zeros(columns, cost.dtype)
    known_u = jnp.arange(rows) == 0
    known_v = jnp.zeros(columns, dtype=jnp.bool_)

    def propagate(
        _: Array, state: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        u, v, known_u, known_v = state
        row_neighbors = basis & known_v[None, :]
        column_for_row = jnp.argmax(row_neighbors, axis=1)
        add_u = ~known_u & jnp.any(row_neighbors, axis=1)
        u = jnp.where(
            add_u, cost[jnp.arange(rows), column_for_row] - v[column_for_row], u
        )
        known_u |= add_u
        column_neighbors = basis & known_u[:, None]
        row_for_column = jnp.argmax(column_neighbors, axis=0)
        add_v = ~known_v & jnp.any(column_neighbors, axis=0)
        v = jnp.where(
            add_v, cost[row_for_column, jnp.arange(columns)] - u[row_for_column], v
        )
        return u, v, known_u, known_v | add_v

    u, v, *_ = jax.lax.fori_loop(0, rows + columns, propagate, (u, v, known_u, known_v))
    return u, v


def _cycle_signs(basis: Array, entering: Array) -> Array:
    """Return the unique alternating cycle formed by one nonbasic edge."""
    rows, columns = basis.shape
    root, target = entering // columns, rows + entering % columns
    nodes = rows + columns
    parents = jnp.full(nodes, -1, dtype=entering.dtype).at[root].set(root)

    def discover(_: Array, parents: Array) -> Array:
        known_rows, known_columns = parents[:rows] >= 0, parents[rows:] >= 0
        row_neighbors = basis & known_columns[None, :]
        column_neighbors = basis & known_rows[:, None]
        row_parent = rows + jnp.argmax(row_neighbors, axis=1)
        column_parent = jnp.argmax(column_neighbors, axis=0)
        add_rows = ~known_rows & jnp.any(row_neighbors, axis=1)
        add_columns = ~known_columns & jnp.any(column_neighbors, axis=0)
        return jnp.concatenate(
            (
                jnp.where(add_rows, row_parent, parents[:rows]),
                jnp.where(add_columns, column_parent, parents[rows:]),
            )
        )

    parents = jax.lax.fori_loop(0, nodes, discover, parents)
    signs = jnp.zeros((rows, columns), dtype=jnp.int32)
    signs = signs.at[root, target - rows].set(1)

    def trace(
        _: Array, state: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        signs, node, sign = state
        parent = parents[node]
        row = jnp.where(node < rows, node, parent)
        column = jnp.where(node >= rows, node - rows, parent - rows)
        # At the root no edge is visited. Clip the unused structural indices
        # before gathering instead of relying on out-of-bounds indexing.
        row = jnp.clip(row, 0, rows - 1)
        column = jnp.clip(column, 0, columns - 1)
        signs = signs.at[row, column].set(
            jnp.where(node == root, signs[row, column], sign)
        )
        return signs, jnp.where(node == root, root, parent), -sign

    signs, *_ = jax.lax.fori_loop(
        0, nodes - 1, trace, (signs, target, jnp.asarray(-1, dtype=jnp.int32))
    )
    return signs


def _pivot(plan: Array, basis: Array, entering: Array) -> tuple[Array, Array]:
    signs = _cycle_signs(basis, entering)
    theta = jnp.min(jnp.where(signs < 0, plan, jnp.inf))
    leaving = jnp.argmax(((signs < 0) & (plan == theta)).reshape(-1))
    plan = plan + theta * signs
    basis = basis.reshape(-1).at[leaving].set(False).at[entering].set(True)
    return plan, basis.reshape(plan.shape)


def _valid_probability(masses: Array) -> Array:
    """Check supplied marginal values at their original floating precision."""
    values = masses.astype(jnp.result_type(masses, 1.0))
    return jnp.all(jnp.isfinite(values) & (values >= 0)) & (
        jnp.abs(values.sum() - 1) <= 8 * jnp.finfo(values.dtype).eps
    )


def _scaled_cost(cost: Array) -> tuple[Array, Array]:
    """Normalize nonnegative costs without forming a subnormal reciprocal."""
    scale = jnp.max(cost)
    scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    # XLA can lower direct division to a subnormal reciprocal and flush even
    # cost == scale to zero near the largest finite float. Dividing bounded
    # mantissas before applying the exponent difference preserves that ratio.
    mantissa, exponent = jnp.frexp(cost)
    scale_mantissa, scale_exponent = jnp.frexp(scale)
    return scale, jnp.ldexp(mantissa / scale_mantissa, exponent - scale_exponent)


def solve_balanced_transport(  # noqa: PLR0914
    cost: Array,
    left_masses: Array,
    right_masses: Array,
    *,
    max_iterations: int = 10_000,
    tolerance: float | Array = 1e-6,
) -> TransportationResult:
    """Solve min sum(plan * cost) under two probability marginals.

    Nonempty rectangular costs and matching vector shapes are structural
    requirements. Costs must be finite and nonnegative; masses must be finite,
    nonnegative, and sum to one. Those numerical laws become ``input_valid``
    evidence under transformations. Invalid values use safe working inputs and
    return a NaN objective. The solver never normalizes supplied marginals.

    ``tolerance`` bounds both the marginal residual and the dual gap divided
    by the largest input cost (one when all costs are zero). It does not relax
    the probability-law checks, which use caller-precision rounding slack.
    The largest cost scales the solve, without changing its minimizing plan.
    """
    cost, a, b = jnp.asarray(cost), jnp.asarray(left_masses), jnp.asarray(right_masses)
    if (
        cost.ndim != _MATRIX_DIMENSIONS
        or not all(cost.shape)
        or a.shape != (cost.shape[0],)
        or b.shape != (cost.shape[1],)
    ):
        message = "Transport requires a nonempty cost matrix and matching marginals"
        raise ValueError(message)
    if type(max_iterations) is not int or max_iterations < 1:
        message = "Transport max_iterations must be a positive integer"
        raise ValueError(message)
    if any(jnp.iscomplexobj(value) for value in (cost, a, b)):
        message = "Transport costs and marginals must be real"
        raise TypeError(message)
    # Probability evidence belongs to each supplied marginal's precision.
    # Valid float32 thirds need not satisfy float64 rounding slack when paired
    # with float64 costs. No renormalization changes their supplied values.
    valid_probabilities = _valid_probability(a) & _valid_probability(b)
    dtype = jnp.result_type(cost, a, b, 1.0)
    cost, a, b = cost.astype(dtype), a.astype(dtype), b.astype(dtype)
    tol = jnp.asarray(tolerance, dtype=dtype)
    if tol.ndim != 0:
        message = "Transport tolerance must be a scalar"
        raise ValueError(message)
    input_valid = (
        jnp.all(jnp.isfinite(cost) & (cost >= 0))
        & valid_probabilities
        & jnp.isfinite(tol)
        & (tol > 0)
    )
    safe_a = jnp.where(input_valid, a, jnp.full_like(a, 1 / a.size))
    safe_b = jnp.where(input_valid, b, jnp.full_like(b, 1 / b.size))
    active_pairs = (safe_a[:, None] > 0) & (safe_b[None, :] > 0)
    safe_cost = jnp.where(input_valid & active_pairs, cost, 0)
    scale, working_cost = _scaled_cost(safe_cost)
    safe_tol = jnp.where(input_valid, tol, jnp.asarray(1e-6, dtype=dtype))
    plan, basis = _north_west_basis(safe_a, safe_b)

    def condition(state: tuple[Array, Array, Array, Array]) -> Array:
        _, _, iteration, keep_going = state
        return keep_going & (iteration < max_iterations)

    def iterate(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        plan, basis, iteration, _ = state
        u, v = _tree_potentials(working_cost, basis)
        eligible = ~basis & (working_cost - u[:, None] - v[None, :] < -safe_tol)
        keep_going = jnp.any(eligible)
        entering = jnp.argmax(eligible.reshape(-1))

        def pivot(pair: tuple[Array, Array]) -> tuple[Array, Array]:
            return _pivot(pair[0], pair[1], entering)

        def retain(pair: tuple[Array, Array]) -> tuple[Array, Array]:
            return pair

        plan, basis = jax.lax.cond(
            keep_going,
            pivot,
            retain,
            (plan, basis),
        )
        return plan, basis, iteration + keep_going, keep_going

    plan, basis, iterations, _ = jax.lax.while_loop(
        condition,
        iterate,
        (plan, basis, jnp.asarray(0, dtype=jnp.int32), jnp.ones((), dtype=jnp.bool_)),
    )
    u, v = _tree_potentials(working_cost, basis)
    dual_violation = jnp.maximum(jnp.max(u[:, None] + v[None, :] - working_cost), 0)
    # A common row shift makes the returned dual feasible even at the stopping
    # tolerance, so its objective is an actual lower bound for these marginals.
    u = u - dual_violation
    objective = jnp.sum(plan * working_cost)
    dual = safe_a @ u + safe_b @ v
    gap = objective - dual
    residual = jnp.maximum(
        jnp.max(jnp.abs(plan.sum(axis=1) - safe_a)),
        jnp.max(jnp.abs(plan.sum(axis=0) - safe_b)),
    )
    finite = (
        jnp.isfinite(objective * scale)
        & jnp.all(jnp.isfinite(plan))
        & jnp.all(jnp.isfinite(u * scale))
        & jnp.all(jnp.isfinite(v * scale))
        & jnp.isfinite(gap * scale)
    )
    converged = (
        input_valid
        & finite
        & (residual <= safe_tol)
        & (jnp.min(plan) >= -safe_tol)
        & (jnp.abs(gap) <= safe_tol)
    )
    return TransportationResult(
        value=jnp.where(converged, objective * scale, jnp.nan),
        plan=plan,
        row_potentials=u * scale,
        column_potentials=v * scale,
        gap=jnp.where(input_valid, gap * scale, jnp.nan),
        residual=jnp.where(input_valid, residual, jnp.nan),
        converged=converged,
        iterations=iterations,
        input_valid=input_valid,
        finite=finite,
    )
