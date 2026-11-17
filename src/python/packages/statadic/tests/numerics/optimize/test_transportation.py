"""Independent primal/dual and transformation witnesses for finite transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog
from statadic.geometry.discrepancy.transport.weighted import (
    balanced_transport_from_cost_result,
)
from statadic.numerics.optimize.transportation import solve_balanced_transport

if TYPE_CHECKING:
    from statadic.core.typing import Array
    from statadic.numerics.optimize.transportation import TransportationResult

_TOLERANCE = 1e-12
_NONZERO_GAP = 1e-3


def _reference(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    rows, columns = cost.shape
    constraints = np.concatenate(
        (
            np.kron(np.eye(rows), np.ones((1, columns))),
            np.tile(np.eye(columns), (1, rows)),
        )
    )
    result = linprog(
        cost.reshape(-1),
        A_eq=constraints,
        b_eq=np.concatenate((a, b)),
        bounds=(0, None),
        method="highs",
    )
    assert result.success
    return float(result.fun)


@pytest.mark.unit
def test_unequal_weighted_plan_retains_primal_and_dual_witnesses() -> None:
    """A rectangular nonuniform problem needs two pivots and attains its dual."""
    cost = jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]])
    a, b = jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])
    result = balanced_transport_from_cost_result(cost, a, b, tolerance=1e-12)
    assert bool(result.input_valid & result.converged & result.finite)
    assert int(result.iterations) >= 1
    assert float(result.value) == pytest.approx(0.7)
    np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=1e-13)
    np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=1e-13)
    assert np.all(np.asarray(result.plan) >= 0)
    assert np.all(
        np.asarray(result.row_potentials[:, None] + result.column_potentials)
        <= np.asarray(cost) + 1e-12
    )
    assert float(jnp.sum(result.plan * cost)) == pytest.approx(float(result.value))
    assert float(a @ result.row_potentials + b @ result.column_potentials) == (
        pytest.approx(float(result.value), abs=1e-12)
    )
    assert float(result.gap) == pytest.approx(0, abs=1e-12)
    assert float(result.residual) <= _TOLERANCE


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (2, 7), (6, 3), (5, 5)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_transportation_matches_independent_linear_program(
    shape: tuple[int, int], dtype: type
) -> None:
    """An independent transportation LP checks costs and nonuniform supports."""
    rng = np.random.default_rng(301 + shape[0] + shape[1])
    cost = rng.uniform(0, 4, size=shape)
    a, b = rng.dirichlet(np.ones(shape[0])), rng.dirichlet(np.ones(shape[1]))
    reference = _reference(cost, a, b)
    tolerance = 4e-6 if dtype == jnp.float32 else 1e-11

    def solve(c: Array, x: Array, y: Array) -> TransportationResult:
        return solve_balanced_transport(c, x, y, tolerance=tolerance)

    result = jax.jit(solve)(
        jnp.asarray(cost, dtype=dtype),
        jnp.asarray(a, dtype=dtype),
        jnp.asarray(b, dtype=dtype),
    )
    assert bool(result.converged)
    assert float(result.value) == pytest.approx(reference, abs=3 * tolerance)
    assert result.value.dtype == dtype


@pytest.mark.unit
def test_zero_mass_storage_and_ties_cannot_change_transport() -> None:
    """Zero probabilities remain inert, including large unused ground costs."""
    cost = jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]])
    a, b = jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])
    padded_cost = jnp.pad(cost, ((1, 1), (1, 1)), constant_values=1e100)
    result = solve_balanced_transport(
        padded_cost, jnp.pad(a, (1, 1)), jnp.pad(b, (1, 1)), tolerance=1e-11
    )
    assert bool(result.converged)
    assert float(result.value) == pytest.approx(0.7, abs=1e-11)
    np.testing.assert_allclose(result.plan[jnp.array([0, -1])], 0, atol=0)
    np.testing.assert_allclose(result.plan[:, jnp.array([0, -1])], 0, atol=0)
    tied = solve_balanced_transport(
        jnp.zeros((4, 7)), jnp.ones(4) / 4, jnp.ones(7) / 7, tolerance=1e-12
    )
    assert bool(tied.converged)
    assert float(tied.value) == pytest.approx(0, abs=0)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("seed", [7, 11, 29])
def test_degenerate_uniform_marginals_and_tied_costs_match_lp(seed: int) -> None:
    """Repeated costs and zero-flow tree edges still attain an independent optimum."""
    rng = np.random.default_rng(seed)
    cost = rng.integers(0, 4, size=(4, 6)).astype(float)
    a, b = np.ones(4) / 4, np.ones(6) / 6
    result = solve_balanced_transport(
        jnp.asarray(cost), jnp.asarray(a), jnp.asarray(b), tolerance=1e-12
    )
    assert bool(result.converged)
    assert float(result.value) == pytest.approx(_reference(cost, a, b), abs=1e-11)
    assert np.all(np.asarray(result.plan) >= 0)


@pytest.mark.unit
@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_positive_cost_scaling_preserves_plan_and_value(scale: float) -> None:
    """Cost normalization retains meaningful very small and very large values."""
    cost = jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]]) * scale
    result = solve_balanced_transport(
        cost, jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5]), tolerance=1e-12
    )
    assert bool(result.converged)
    assert float(result.value / scale) == pytest.approx(0.7)


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("compiled", [False, True])
def test_largest_finite_constant_cost_retains_its_transport_value(
    dtype: type, *, compiled: bool
) -> None:
    """A constant largest-finite objective must not vanish through its reciprocal."""
    maximum = jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)
    cost = jnp.full((2, 2), maximum, dtype=dtype)
    masses = jnp.array([0.5, 0.5], dtype=dtype)

    def solve(c: Array, a: Array) -> TransportationResult:
        return solve_balanced_transport(c, a, a)

    evaluate = jax.jit(solve) if compiled else solve
    result = evaluate(cost, masses)
    assert bool(result.input_valid & result.finite & result.converged)
    assert result.value.dtype == dtype
    np.testing.assert_array_equal(result.value, maximum)
    np.testing.assert_array_equal(result.gap, 0)
    np.testing.assert_array_equal(result.residual, 0)


@pytest.mark.unit
def test_nested_jit_and_vmap_retain_separate_solution_evidence() -> None:
    """Batch lanes keep their own plans, convergence and dual evidence."""
    costs = jnp.array(
        [
            [[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]],
            [[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]],
        ]
    )
    a, b = jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])

    def solve(c: Array) -> TransportationResult:
        return solve_balanced_transport(c, a, b, tolerance=1e-12)

    evaluate = jax.jit(solve)
    results = jax.jit(jax.vmap(evaluate))(costs)
    assert results.plan.shape == (2, 2, 3)
    assert bool(jnp.all(results.converged))
    for index in range(2):
        assert float(results.value[index]) == pytest.approx(
            _reference(np.asarray(costs[index]), np.asarray(a), np.asarray(b))
        )


@pytest.mark.unit
def test_iteration_exhaustion_does_not_expose_a_partial_fit_as_optimal() -> None:
    """A feasible but suboptimal first pivot remains unavailable to consumers."""
    result = solve_balanced_transport(
        jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]]),
        jnp.array([0.4, 0.6]),
        jnp.array([0.2, 0.3, 0.5]),
        max_iterations=1,
        tolerance=1e-12,
    )
    assert bool(result.input_valid & result.finite)
    assert not bool(result.converged)
    assert bool(jnp.isnan(result.value))
    assert float(result.residual) <= _TOLERANCE
    assert float(result.gap) > _NONZERO_GAP


@pytest.mark.unit
@pytest.mark.parametrize("invalid", ["cost", "negative", "mass", "tolerance"])
def test_invalid_numeric_inputs_stay_visible_under_jit(invalid: str) -> None:
    """Probability-domain failures return unavailable lanes under transforms."""
    cost = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    a, b, tolerance = jnp.array([0.4, 0.6]), jnp.array([0.3, 0.7]), 1e-6
    if invalid == "cost":
        cost = cost.at[0, 0].set(jnp.nan)
    elif invalid == "negative":
        cost = cost.at[0, 0].set(-1)
    elif invalid == "mass":
        a = a * 2
    else:
        tolerance = -1.0

    def solve(c: Array, x: Array, y: Array) -> TransportationResult:
        return solve_balanced_transport(c, x, y, tolerance=tolerance)

    result = jax.jit(solve)(cost, a, b)
    assert not bool(result.input_valid)
    assert not bool(result.converged)
    assert bool(jnp.isnan(result.value))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_rounding_slack_does_not_admit_unbalanced_total_mass(dtype: type) -> None:
    """A total-mass change larger than the probability contract is invalid."""
    a = jnp.array([0.4, 0.6], dtype=dtype)
    b = jnp.array([0.2, 0.3, 0.5], dtype=dtype)
    b = b.at[-1].add(32 * jnp.finfo(dtype).eps)
    result = solve_balanced_transport(jnp.ones((2, 3), dtype=dtype), a, b)
    assert not bool(result.input_valid)
    assert not bool(result.converged)
    assert bool(jnp.isnan(result.value))


@pytest.mark.unit
@pytest.mark.reference
def test_original_mass_precision_survives_higher_precision_costs() -> None:
    """Valid float32 thirds retain their values beside float64 ground costs."""
    cost = jnp.array(
        [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]], dtype=jnp.float64
    )
    masses = jnp.full(3, 1 / 3, dtype=jnp.float32)

    def solve(c: Array, a: Array) -> TransportationResult:
        return solve_balanced_transport(c, a, a, tolerance=1e-12)

    result = jax.jit(solve)(cost, masses)
    expected = _reference(np.asarray(cost), np.asarray(masses), np.asarray(masses))
    assert bool(result.input_valid & result.converged & result.finite)
    assert result.value.dtype == jnp.float64
    assert float(result.value) == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(
        result.plan.sum(axis=1), masses.astype(jnp.float64), rtol=0, atol=1e-12
    )
