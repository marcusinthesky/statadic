# jaxtyping shape strings are runtime contracts.
"""Transform and parity gates for maintained external JAX solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.results import (
    SolverStatus,
)
from statadic.optimize.backends import (
    _newton_root_result,
    bounded_scalar_minimise,
    least_squares,
    least_squares_result,
    minimise_bfgs,
    minimise_bfgs_result,
)

from .._budgets import examples

if TYPE_CHECKING:
    from jax import Array

if TYPE_CHECKING:
    from statadic.core.results import (
        SolverResult,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_newton_root_result_retains_success_and_failures_under_jit() -> None:
    """Root termination and nonfinite lanes keep distinct numerical statuses."""
    target = jnp.array([1.0, -2.0])
    tolerance = 1e-6
    budget = 8

    def solve(initial: jax.Array, budget: int) -> SolverResult[jax.Array]:
        return _newton_root_result(
            lambda value: value - target,
            initial,
            max_steps=budget,
            rtol=tolerance,
            atol=tolerance,
        )

    def solve_with_budget(value: jax.Array) -> SolverResult[jax.Array]:
        return solve(value, budget)

    result = jax.jit(solve_with_budget)(jnp.zeros(2))
    assert bool(result.converged)
    assert int(result.status) == SolverStatus.SUCCESS
    np.testing.assert_allclose(result.value, target, atol=tolerance)
    assert float(result.residual_norm) < tolerance
    assert int(result.rank) == -1
    assert int(result.max_steps) == budget

    exhausted = solve(jnp.zeros(2), 0)
    assert not bool(exhausted.converged)
    assert int(exhausted.status) == SolverStatus.MAX_STEPS
    assert int(exhausted.iterations) == 0

    nonfinite = solve(jnp.array([jnp.nan, 0.0]), budget)
    assert not bool(nonfinite.converged)
    assert int(nonfinite.status) == SolverStatus.NONFINITE


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("grid_points", 0),
        ("grid_points", 1),
        ("grid_points", -1),
        ("grid_points", True),
        ("grid_points", 2.5),
        ("max_steps", -1),
        ("max_steps", False),
        ("max_steps", 1.5),
    ],
)
def test_bounded_scalar_rejects_invalid_static_controls(
    option: str, value: float
) -> None:
    """The grid and iteration budget must be well-defined static integers."""
    with pytest.raises((ValueError, TypeError)):  # noqa: PT012
        if option == "grid_points":
            bounded_scalar_minimise(
                lambda x: (x - 0.37) ** 2,
                jnp.array([0.0, 1.0]),
                grid_points=cast("int", value),
            )
        else:
            bounded_scalar_minimise(
                lambda x: (x - 0.37) ** 2,
                jnp.array([0.0, 1.0]),
                max_steps=cast("int", value),
            )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "bounds",
    [
        jnp.array([0.0]),
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([[0.0, 1.0]]),
        jnp.array([0.0 + 1j, 1.0 + 0j]),
        jnp.array([False, True]),
    ],
)
def test_bounded_scalar_rejects_invalid_bound_structure(bounds: jax.Array) -> None:
    """Bounds must be a real numeric pair."""
    with pytest.raises((ValueError, TypeError)):
        bounded_scalar_minimise(lambda x: x**2, bounds)


@pytest.mark.component
@pytest.mark.contract
def test_bounded_scalar_rejects_invalid_tolerance_structure() -> None:
    """Tolerance has a real scalar structure."""
    for tolerance in (jnp.array([1.0]), jnp.array(1j), jnp.ones((), dtype=bool)):
        with pytest.raises((ValueError, TypeError)):
            bounded_scalar_minimise(
                lambda x: x**2, jnp.array([0.0, 1.0]), tolerance=tolerance
            )


@pytest.mark.component
@pytest.mark.contract
def test_bounded_scalar_mixed_dynamic_input_lanes() -> None:
    """Invalid traced bounds and tolerances cannot contaminate a valid lane."""
    bounds = jnp.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [jnp.nan, 1.0],
            [0.0, jnp.inf],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    tolerance = jnp.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, -1.0, 0.0, jnp.nan, jnp.inf])

    def solve(b: jax.Array, tol: jax.Array) -> SolverResult[jax.Array]:
        return bounded_scalar_minimise(lambda x: (x - 0.37) ** 2, b, tolerance=tol)

    result = jax.jit(jax.vmap(solve))(bounds, tolerance)
    np.testing.assert_array_equal(result.converged, [True] + [False] * 8)
    np.testing.assert_array_equal(
        result.status,
        [
            SolverStatus.SUCCESS,
            SolverStatus.FAILED,
            SolverStatus.FAILED,
            SolverStatus.NONFINITE,
            SolverStatus.NONFINITE,
            SolverStatus.FAILED,
            SolverStatus.FAILED,
            SolverStatus.NONFINITE,
            SolverStatus.NONFINITE,
        ],
    )
    np.testing.assert_allclose(result.value[0], [0.37], atol=1e-5)
    assert float(result.residual_norm[0]) >= 0
    assert bool(jnp.all(jnp.isnan(result.value[1:])))
    assert bool(jnp.all(jnp.isnan(result.residual_norm[1:])))


@pytest.mark.component
@pytest.mark.contract
def test_bounded_scalar_budget_and_nonfinite_objectives() -> None:
    """Finite sampled candidates survive partial domain failure, not total failure."""
    bounds = jnp.array([0.0, 1.0])
    exhausted = bounded_scalar_minimise(lambda x: (x - 0.37) ** 2, bounds, max_steps=0)
    assert not bool(exhausted.converged)
    assert int(exhausted.status) == SolverStatus.MAX_STEPS
    assert float(exhausted.residual_norm) > 0
    lower_domain = 0.2

    def evaluate_bounded_scalar_minimise(b: jax.Array) -> SolverResult[Array]:
        return bounded_scalar_minimise(
            lambda x: jnp.where(x < lower_domain, jnp.nan, (x - 0.37) ** 2), b
        )

    partial = jax.jit(evaluate_bounded_scalar_minimise)(bounds)
    assert bool(partial.converged)
    np.testing.assert_allclose(partial.value, [0.37], atol=1e-5)
    for bad in (jnp.nan, jnp.inf, -jnp.inf):
        result = bounded_scalar_minimise(
            lambda x, bad=bad: jnp.full_like(x, bad), bounds
        )
        assert not bool(result.converged)
        assert int(result.status) == SolverStatus.NONFINITE
        assert bool(jnp.isnan(result.value[0]))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bounded_scalar_rejects_unrepresentable_width(dtype: type) -> None:
    """Finite endpoints alone do not guarantee a representable interval width."""
    limit = np.finfo(dtype).max

    def evaluate_bounded_scalar_minimise(b: jax.Array) -> SolverResult[Array]:
        return bounded_scalar_minimise(lambda x: x**2, b)

    with jax.enable_x64(new_val=dtype is np.float64):
        result = jax.jit(evaluate_bounded_scalar_minimise)(
            jnp.asarray([-limit, limit], dtype=dtype)
        )
    assert not bool(result.converged)
    assert int(result.status) == SolverStatus.NONFINITE
    assert bool(jnp.isnan(result.value[0]))
    assert bool(jnp.isnan(result.residual_norm))


@pytest.mark.component
@pytest.mark.contract
def test_bounded_scalar_promotes_strong_tolerance_before_grid() -> None:
    """A higher-precision tolerance controls the grid and its epsilon floor."""

    def solve(b: jax.Array, tol: jax.Array) -> SolverResult[jax.Array]:
        return bounded_scalar_minimise(lambda x: (x - center) ** 2, b, tolerance=tol)

    def first_coordinate(a: jax.Array) -> Array:
        return a[0]

    with jax.enable_x64(new_val=True):
        bounds = jnp.array([0, 1], dtype=jnp.float32)
        tolerance = jnp.array(1e-12, dtype=jnp.float64)
        center = jnp.array(0.123456789, dtype=jnp.float64)

        eager = solve(bounds, tolerance)
        compiled = jax.jit(solve)(bounds, tolerance)
        batched = jax.jit(jax.vmap(solve))(
            jnp.stack([bounds, bounds]), jnp.stack([tolerance, tolerance])
        )

        for result in (eager, compiled, jax.tree.map(first_coordinate, batched)):
            assert bool(result.converged)
            assert result.value.dtype == jnp.float64
            assert float(result.residual_norm) <= float(tolerance)
            np.testing.assert_allclose(result.value, [center], rtol=0, atol=1e-11)


@pytest.mark.component
@pytest.mark.numerical
@examples(20)
@example(center=-0.4)
@example(center=0.0)
@example(center=0.7)
@example(center=1.4)
@given(center=st.floats(-2.0, 3.0, width=32))
def test_bounded_scalar_cached_bracket_matches_clipped_quadratic(center: float) -> None:
    """Interior and boundary minima retain their bracket and objective semantics."""

    def evaluate_bounded_scalar_minimise(c: jax.Array) -> SolverResult[Array]:
        return bounded_scalar_minimise(lambda x: (x - c) ** 2, jnp.array([0.0, 1.0]))

    result = jax.jit(evaluate_bounded_scalar_minimise)(jnp.asarray(center))
    assert bool(result.converged)
    assert int(result.status) == SolverStatus.SUCCESS
    assert float(result.residual_norm) >= 0
    np.testing.assert_allclose(result.value, [np.clip(center, 0, 1)], rtol=0, atol=1e-5)


@pytest.mark.component
@pytest.mark.contract
def test_bounded_scalar_flat_objective_retains_valid_grid_candidate() -> None:
    """Ties remain finite and converged without requiring a unique minimizer."""
    result = bounded_scalar_minimise(jnp.ones_like, jnp.array([0.0, 1.0]))
    assert bool(result.converged)
    assert 0 <= float(result.value[0]) <= 1
    tolerance = 1e-6
    assert 0 <= float(result.residual_norm) < tolerance


@pytest.mark.unit
@pytest.mark.reference
def test_qr_least_squares_matches_numpy_and_transforms() -> None:
    """Match float64 QR coefficients under eager, jit, and batched outcomes."""
    rng = np.random.default_rng(4)
    design_np = rng.normal(size=(20, 4))
    outcomes_np = rng.normal(size=(3, 20))
    expected = np.stack(
        [np.linalg.lstsq(design_np, outcome, rcond=None)[0] for outcome in outcomes_np]
    )
    with jax.enable_x64(new_val=True):
        design = jnp.asarray(design_np)
        outcomes = jnp.asarray(outcomes_np)
        eager = least_squares(design, outcomes[0])
        compiled = jax.jit(least_squares)(design, outcomes[0])
        batched = jax.vmap(least_squares, in_axes=(None, 0))(design, outcomes)
    np.testing.assert_allclose(np.asarray(eager), expected[0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(compiled), expected[0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(batched), expected, atol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("shape", [(9, 3), (3, 9), (3, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qr_preserves_full_rank_least_squares_and_minimum_norm(
    shape: tuple[int, int], dtype: type[np.float32 | np.float64]
) -> None:
    """Tall, square and wide QR use the same full-rank mathematical solution."""
    rng = np.random.default_rng(37)
    design_np = rng.normal(size=shape).astype(dtype)
    outcomes_np = rng.normal(size=(2, shape[0])).astype(dtype)
    expected = np.stack(
        [np.linalg.lstsq(design_np, outcome, rcond=None)[0] for outcome in outcomes_np]
    )
    # Forward coefficient error scales with conditioning; the square fixture
    # has condition number about 107, while its rank remains well identified.
    tolerance = max(
        1e-12,
        4 * np.finfo(dtype).eps * np.linalg.cond(design_np.astype(np.float64)),
    )
    with jax.enable_x64(new_val=dtype is np.float64):
        design, outcomes = jnp.asarray(design_np), jnp.asarray(outcomes_np)
        result = jax.jit(jax.vmap(least_squares_result, in_axes=(None, 0)))(
            design, outcomes
        )
        values = jax.jit(jax.vmap(least_squares, in_axes=(None, 0)))(design, outcomes)
    assert result.value.dtype == dtype
    np.testing.assert_allclose(result.value, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(result.rank, min(shape))
    assert bool(jnp.all(result.converged))
    np.testing.assert_array_equal(result.status, SolverStatus.SUCCESS)
    np.testing.assert_array_equal(result.iterations, 0)
    np.testing.assert_array_equal(result.max_steps, 0)
    if shape[0] < shape[1]:
        _, _, vh = np.linalg.svd(design_np.astype(np.float64), full_matrices=True)
        null_component = np.asarray(result.value) @ vh[shape[0] :].T
        np.testing.assert_allclose(null_component, 0, atol=tolerance)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("shape", [(9, 3), (3, 9)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qr_jacobians_preserve_response_and_design_derivatives(
    shape: tuple[int, int], dtype: type[np.float32 | np.float64]
) -> None:
    """Full-rank response Jacobians and design derivatives match independent math."""
    rng = np.random.default_rng(17)
    design_np = rng.normal(size=shape).astype(dtype)
    outcome_np = rng.normal(size=shape[0]).astype(dtype)
    direction = rng.normal(size=shape)
    contrast = rng.normal(size=shape[1])
    inverse = np.linalg.lstsq(
        design_np.astype(np.float64), np.eye(shape[0]), rcond=None
    )[0]
    step = 1e-5
    expected_directional = (
        (
            np.linalg.lstsq(design_np + step * direction, outcome_np, rcond=None)[0]
            - np.linalg.lstsq(design_np - step * direction, outcome_np, rcond=None)[0]
        )
        @ contrast
        / (2 * step)
    )
    tolerance = 3e-5 if dtype is np.float32 else 1e-8

    def objective(matrix: jax.Array) -> jax.Array:
        return least_squares_result(matrix, outcome).value @ jnp.asarray(
            contrast, dtype=matrix.dtype
        )

    with jax.enable_x64(new_val=dtype is np.float64):
        design, outcome = jnp.asarray(design_np), jnp.asarray(outcome_np)
        forward = jax.jit(jax.jacfwd(least_squares, argnums=1))(design, outcome)
        reverse = jax.jit(jax.jacrev(least_squares, argnums=1))(design, outcome)

        gradient = jax.jit(jax.grad(objective))(design)
    np.testing.assert_allclose(forward, inverse, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(reverse, inverse, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        np.sum(np.asarray(gradient) * direction),
        expected_directional,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_qr_value_door_preserves_runtime_failure_behavior() -> None:
    """The value API still raises when its nonthrowing result retains bad output."""
    design = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    outcome = jnp.asarray([jnp.nan, 2.0, 4.0])
    result = least_squares_result(design, outcome)
    assert not result.converged
    assert int(result.status) == SolverStatus.NONFINITE
    with pytest.raises(RuntimeError, match="non-finite"):
        least_squares(design, outcome)
    with pytest.raises(RuntimeError, match="non-finite"):
        jax.jit(least_squares)(design, outcome)


@pytest.mark.integration
@pytest.mark.contract
def test_qr_preserves_matching_dtype_requirement() -> None:
    """A backend switch cannot silently promote a previously rejected response."""
    with jax.enable_x64(new_val=True):
        design = jnp.ones((3, 2), dtype=jnp.float64)
        outcome = jnp.ones(3, dtype=jnp.float32)
        for solve in (least_squares, least_squares_result):
            with pytest.raises(ValueError, match="structures do not match"):
                solve(design, outcome)


@pytest.mark.integration
@pytest.mark.numerical
def test_optimistix_bfgs_minimises_a_typed_quadratic_under_jit() -> None:
    """Exercise the nonlinear adapter without claiming GEL-equivalent semantics."""
    target = jnp.array([1.5, -2.0])

    def objective(value: jax.Array) -> jax.Array:
        return jnp.sum((value - target) ** 2)

    initial = jnp.zeros(2)
    eager = minimise_bfgs(objective, initial)

    def evaluate_minimise_bfgs(value: jax.Array) -> Array:
        return minimise_bfgs(objective, value)

    compiled = jax.jit(evaluate_minimise_bfgs)(initial)
    assert np.asarray(eager) == pytest.approx(np.asarray(target), abs=1e-6)
    assert np.asarray(compiled) == pytest.approx(np.asarray(target), abs=1e-6)


@pytest.mark.integration
@pytest.mark.contract
def test_qr_result_preserves_the_iterate_and_separates_rank() -> None:
    """Backend success on a finite singular QR solve does not prove identification."""
    design = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    outcome = jnp.array([1.0, 2.0, 4.0])
    result = jax.jit(least_squares_result)(design, outcome)
    expected = np.linalg.lstsq(np.asarray(design), np.asarray(outcome), rcond=None)[0]
    np.testing.assert_allclose(result.value, expected, atol=1e-12)
    np.testing.assert_array_equal(result.value, least_squares(design, outcome))
    assert bool(result.converged)
    assert int(result.status) == SolverStatus.SUCCESS
    assert int(result.rank) == design.shape[1]
    assert int(result.iterations) == int(result.max_steps) == 0
    tolerance = 1e-12
    assert float(result.residual_norm) < tolerance

    singular = least_squares_result(jnp.ones((3, 2)), jnp.ones(3))
    assert int(singular.rank) == 1
    # A singular triangular solve may yield finite or nonfinite output by backend.
    finite = np.all(np.isfinite(singular.value)) and np.isfinite(singular.residual_norm)
    assert bool(singular.converged) == finite
    assert int(singular.status) == (
        SolverStatus.SUCCESS if finite else SolverStatus.NONFINITE
    )


@pytest.mark.integration
@pytest.mark.contract
def test_qr_result_retains_mixed_lane_failure_without_throwing() -> None:
    """A nonfinite lane cannot erase a valid lane's diagnostics under jit/vmap."""
    design = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    outcomes = jnp.array([[1.0, 2.0, 4.0], [jnp.nan, 2.0, 4.0]])
    result = jax.jit(jax.vmap(least_squares_result, in_axes=(None, 0)))(
        design, outcomes
    )
    np.testing.assert_array_equal(result.converged, [True, False])
    np.testing.assert_array_equal(
        result.status, [SolverStatus.SUCCESS, SolverStatus.NONFINITE]
    )
    np.testing.assert_array_equal(result.rank, [2, -1])
    assert result.value.shape == (2, 2)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(result))


@pytest.mark.integration
@pytest.mark.contract
def test_bfgs_result_keeps_budget_exhaustion_per_lane() -> None:
    """A converged initial point and an exhausted lane retain separate statuses."""
    target = jnp.array([1.5, -2.0])

    def objective(value: jax.Array) -> jax.Array:
        return jnp.sum((value - target) ** 2)

    budget = 2

    def solve(initial: jax.Array) -> SolverResult[jax.Array]:
        return minimise_bfgs_result(objective, initial, max_steps=budget)

    result = jax.jit(jax.vmap(solve))(jnp.stack([target, jnp.zeros(2)]))
    np.testing.assert_array_equal(result.converged, [True, False])
    np.testing.assert_array_equal(
        result.status, [SolverStatus.SUCCESS, SolverStatus.MAX_STEPS]
    )
    assert int(result.iterations[1]) == int(result.max_steps[1]) == budget
    np.testing.assert_array_equal(result.residual_norm[0], 0.0)
    assert float(result.residual_norm[1]) > 1.0
    np.testing.assert_array_equal(result.rank, [-1, -1])

    completed = minimise_bfgs_result(objective, jnp.zeros(2))
    np.testing.assert_allclose(completed.value, target, atol=1e-8)
    np.testing.assert_array_equal(
        completed.value, minimise_bfgs(objective, jnp.zeros(2))
    )
    assert bool(completed.converged)
    tolerance = 1e-8
    assert float(completed.residual_norm) < tolerance


@pytest.mark.integration
@pytest.mark.contract
def test_bfgs_result_reports_nonfinite_and_preserves_throwing_wrapper() -> None:
    """The new result door reports failure while the existing door still raises."""

    def objective(value: jax.Array) -> jax.Array:
        return jnp.sum(value**2)

    def evaluate_minimise_bfgs_result(initial: jax.Array) -> SolverResult[Array]:
        return minimise_bfgs_result(objective, initial)

    result = jax.jit(evaluate_minimise_bfgs_result)(jnp.array([jnp.nan, 0.0]))
    assert not bool(result.converged)
    assert int(result.status) == SolverStatus.NONFINITE
    exhausted = minimise_bfgs_result(objective, jnp.ones(2), max_steps=0)
    assert int(exhausted.status) == SolverStatus.MAX_STEPS
    assert int(exhausted.iterations) == 0
    with pytest.raises(RuntimeError, match="maximum number of steps"):
        minimise_bfgs(objective, jnp.ones(2), max_steps=0)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("shape", [(7, 3), (3, 7), (3, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@examples(8)
@given(spectrum=st.lists(st.floats(0.25, 4.0), min_size=3, max_size=3))
def test_qr_known_spectra_preserve_solution_rank_and_backward_residual(
    shape: tuple[int, int], dtype: type[np.float32 | np.float64], spectrum: list[float]
) -> None:
    """Construct identified spectra rather than filtering arbitrary random ranks."""
    generator = np.random.default_rng(1201)
    left = np.linalg.qr(generator.normal(size=(shape[0], 3)))[0]
    right = np.linalg.qr(generator.normal(size=(shape[1], 3)))[0]
    singular = np.asarray(spectrum)
    design = ((left * singular) @ right.T).astype(dtype)
    outcome = np.linspace(-1.0, 2.0, shape[0]).astype(dtype)
    expected = right @ ((left.T @ outcome) / singular)
    result = jax.jit(least_squares_result)(jnp.asarray(design), jnp.asarray(outcome))
    actual = np.asarray(result.value, dtype=np.float64)
    observation_residual = design.astype(np.float64) @ actual - outcome
    normal_residual = design.astype(np.float64).T @ observation_residual
    scale = np.linalg.norm(design) * (
        np.linalg.norm(design) * np.linalg.norm(actual) + np.linalg.norm(outcome)
    )
    margin = 256.0 * float(np.finfo(dtype).eps)

    assert bool(result.converged)
    assert int(result.rank) == min(shape)
    assert result.value.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=margin, atol=margin)
    assert np.linalg.norm(normal_residual, ord=np.inf) / scale < margin
    assert float(result.residual_norm) == pytest.approx(
        float(np.max(np.abs(normal_residual))), abs=margin * scale
    )
    if shape[0] < shape[1]:
        np.testing.assert_allclose(actual, right @ (right.T @ actual), atol=margin)


@pytest.mark.integration
@pytest.mark.numerical
def test_bfgs_coupled_quadratic_checks_objective_and_gradient_independently() -> None:
    """A nonseparable SPD optimum tests value quality beyond backend termination."""
    quadratic = np.asarray([[3.0, 0.4, -0.2], [0.4, 2.0, 0.3], [-0.2, 0.3, 1.0]])
    linear = np.asarray([-1.0, 2.0, -0.5])
    optimum = np.linalg.solve(quadratic, -linear)

    def objective(value: jax.Array) -> jax.Array:
        return (
            0.5 * value @ jnp.asarray(quadratic) @ value + jnp.asarray(linear) @ value
        )

    def evaluate_minimise_bfgs_result(start: jax.Array) -> SolverResult[Array]:
        return minimise_bfgs_result(objective, start)

    result = jax.jit(evaluate_minimise_bfgs_result)(jnp.asarray([3.0, -4.0, 2.0]))
    value = np.asarray(result.value)
    gradient = quadratic @ value + linear
    tolerance = 2e-7
    assert bool(result.converged)
    np.testing.assert_allclose(value, optimum, atol=tolerance, rtol=tolerance)
    assert np.linalg.norm(gradient, ord=np.inf) < tolerance
    assert float(result.residual_norm) == pytest.approx(
        float(np.linalg.norm(gradient, ord=np.inf)), abs=1e-14
    )
    assert float(objective(result.value)) == pytest.approx(
        float(0.5 * optimum @ quadratic @ optimum + linear @ optimum), abs=1e-12
    )


@pytest.mark.integration
@pytest.mark.contract
def test_bfgs_explicit_dynamic_arguments_survive_jit_and_vmap() -> None:
    """The same compiled solver accepts changing data without closures."""
    from statadic.optimize.backends import minimise_bfgs_result_args  # noqa: PLC0415

    def objective(value: jax.Array, target: jax.Array) -> jax.Array:
        return jnp.sum((value - target) ** 2)

    def solve(target: jax.Array) -> object:
        return minimise_bfgs_result_args(objective, jnp.zeros(2), target)

    targets = jnp.array([[1.0, -2.0], [3.0, 4.0]])
    result = jax.jit(jax.vmap(solve))(targets)
    np.testing.assert_allclose(result.value, targets, atol=1e-6)
    assert np.all(np.asarray(result.converged))


@pytest.mark.integration
@pytest.mark.property
def test_sequential_scalar_grid_matches_vectorized_under_jit() -> None:
    """The memory-saving scan preserves bounded search results under JIT."""

    def solve(
        center: jax.Array, strategy: Literal["vectorized", "sequential"]
    ) -> SolverResult[jax.Array]:
        return bounded_scalar_minimise(
            lambda x: (x - center) ** 2,
            jnp.array([-1.0, 1.0]),
            grid_strategy=strategy,
        )

    def evaluate_solve_2(x: jax.Array) -> SolverResult[Array]:
        return solve(x, "sequential")

    sequential = jax.jit(evaluate_solve_2)(jnp.array(0.37))

    def evaluate_solve_3(x: jax.Array) -> SolverResult[Array]:
        return solve(x, "vectorized")

    vectorized = jax.jit(evaluate_solve_3)(jnp.array(0.37))
    np.testing.assert_allclose(sequential.value, vectorized.value)
    assert bool(sequential.converged) == bool(vectorized.converged)


@pytest.mark.component
@pytest.mark.numerical
def test_bounded_scalar_inactive_tail_remains_reverse_differentiable() -> None:
    """Translating both search interval and objective translates the selected point."""

    def location(shift: jax.Array) -> jax.Array:
        return bounded_scalar_minimise(
            lambda x: (x - shift - 0.37) ** 2,
            jnp.array([0.0, 1.0]) + shift,
            max_steps=128,
            tolerance=1e-7,
        ).value[0]

    value, slope = jax.jit(jax.value_and_grad(location))(jnp.asarray(2.0))
    np.testing.assert_allclose(value, 2.37, atol=1e-6)
    np.testing.assert_allclose(slope, 1.0, atol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("wide", [False, True])
def test_qr_rank_uses_original_design_dimension(*, wide: bool) -> None:
    """Tall and wide factorizations must retain the original SVD cutoff."""
    design = np.zeros((64, 2), dtype=np.float32)
    design[0, 0] = 1
    design[1, 1] = 8 * np.finfo(design.dtype).eps
    if wide:
        design = design.T
    answer = least_squares_result(
        jnp.asarray(design), jnp.ones(design.shape[0], dtype=jnp.float32)
    )
    assert int(answer.rank) == np.linalg.matrix_rank(design)
