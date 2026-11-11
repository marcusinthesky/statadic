"""Independent stationarity evidence for the compatibility affine-PGD iterate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.results import SolverStatus
from statadic.optimize.simplex import pgd_simplex_affine, pgd_simplex_affine_result

from .._budgets import examples

if TYPE_CHECKING:
    from statadic.optimize.simplex import SimplexSolution


@pytest.mark.component
@pytest.mark.numerical
def test_affine_result_matches_an_independent_interior_optimum() -> None:
    """Inverse diagonal curvatures give [4/5, 1/5], including objective and KKT."""
    quadratic = jnp.diag(jnp.array([0.25, 1.0]))
    linear = jnp.zeros(2)
    tolerance = 1e-10
    result = pgd_simplex_affine_result(quadratic, linear, tol=tolerance)
    np.testing.assert_allclose(result.solver.value, [0.8, 0.2], atol=3e-10)
    np.testing.assert_array_equal(
        result.solver.value, pgd_simplex_affine(quadratic, linear, tol=1e-10)
    )
    assert float(result.objective) == pytest.approx(0.1, abs=1e-12)
    assert bool(result.feasible)
    assert bool(result.input_finite)
    assert bool(result.convex)
    assert bool(result.solver.converged)
    assert int(result.solver.status) == SolverStatus.SUCCESS
    assert float(result.solver.residual_norm) <= tolerance
    assert 1 < int(result.solver.iterations) < int(result.solver.max_steps)


@pytest.mark.component
@pytest.mark.contract
def test_affine_result_distinguishes_small_steps_from_stationarity() -> None:
    """Large constant simplex curvature makes a tiny step far from the optimum."""
    quadratic = 1e10 * jnp.ones((2, 2)) + jnp.diag(jnp.array([1.0, 2.0]))
    tolerance = 1e-6
    material_error = 0.1
    result = pgd_simplex_affine_result(quadratic, jnp.zeros(2), tol=tolerance)
    assert bool(result.feasible)
    assert bool(result.convex)
    assert int(result.solver.iterations) == 1
    assert float(result.step_norm) < tolerance
    assert float(result.solver.residual_norm) > material_error
    assert not bool(result.solver.converged)
    assert int(result.solver.status) == SolverStatus.STALLED
    # The true optimizer is [2/3, 1/3]; feasibility alone would miss this error.
    assert abs(float(result.solver.value[0]) - 2.0 / 3.0) > material_error


@pytest.mark.component
@pytest.mark.contract
def test_affine_result_preserves_mixed_statuses_under_jit_and_vmap() -> None:
    """Optimum, exhausted budget, invalid PSD and nonfinite lanes stay distinct."""
    quadratics = jnp.stack(
        [
            jnp.eye(2),
            jnp.diag(jnp.array([0.1, 1.0])),
            jnp.diag(jnp.array([-1.0, 1.0])),
            jnp.array([[jnp.nan, 0.0], [0.0, 1.0]]),
        ]
    )
    linear = jnp.zeros((4, 2))

    def evaluate_pgd_simplex_affine_result(
        q: jax.Array, c: jax.Array
    ) -> SimplexSolution:
        return pgd_simplex_affine_result(q, c, maxiter=1)

    result = jax.jit(jax.vmap(evaluate_pgd_simplex_affine_result))(quadratics, linear)
    np.testing.assert_array_equal(result.solver.converged, [True, False, False, False])
    np.testing.assert_array_equal(
        result.solver.status,
        [
            SolverStatus.SUCCESS,
            SolverStatus.MAX_STEPS,
            SolverStatus.FAILED,
            SolverStatus.NONFINITE,
        ],
    )
    assert bool(result.feasible[1])
    material_residual = 1e-3
    assert float(result.solver.residual_norm[1]) > material_residual
    np.testing.assert_array_equal(result.solver.iterations, [1, 1, 1, 1])
    np.testing.assert_array_equal(result.solver.max_steps, [1, 1, 1, 1])
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(result))


@pytest.mark.component
@pytest.mark.contract
def test_affine_result_rejects_nonsymmetric_quadratic_as_a_failed_lane() -> None:
    """Eigenvalues of one triangle cannot certify the stated symmetric-PSD input."""
    result = pgd_simplex_affine_result(
        jnp.array([[1.0, 0.5], [0.0, 1.0]]), jnp.zeros(2)
    )
    assert bool(result.input_finite)
    assert not bool(result.convex)
    assert not bool(result.solver.converged)
    assert int(result.solver.status) == SolverStatus.FAILED


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("enable_x64", [False, True])
def test_affine_result_respects_caller_precision(enable_x64: bool) -> None:  # noqa: FBT001  # pytest parameter
    """Both precisions retain the exact corner without configuration mutation."""
    previous = jax.config.x64_enabled
    with jax.enable_x64(enable_x64):
        quadratic = jnp.eye(2)
        result = pgd_simplex_affine_result(quadratic, jnp.array([10.0, 0.0]))
        expected_dtype = jnp.float64 if enable_x64 else jnp.float32
        assert result.solver.value.dtype == expected_dtype
        np.testing.assert_array_equal(result.solver.value, [0.0, 1.0])
        assert bool(result.solver.converged)
        assert bool(result.feasible)
        assert jax.config.x64_enabled == enable_x64
    assert jax.config.x64_enabled == previous


@pytest.mark.component
@pytest.mark.contract
def test_affine_result_retains_the_reverse_mode_limitation() -> None:
    """A result wrapper does not turn a dynamic while loop into a derivative solver."""
    quadratic = jnp.diag(jnp.array([0.25, 1.0]))

    def differentiated(cost: jax.Array) -> jax.Array:
        result = pgd_simplex_affine_result(quadratic, cost)
        return result.solver.value @ jnp.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Reverse-mode differentiation"):
        jax.grad(differentiated)(jnp.zeros(2))


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(maxiter=0)
@example(maxiter=-1)
@example(maxiter=True)
@given(maxiter=st.one_of(st.integers(max_value=0), st.just(value=True)))
def test_affine_result_rejects_invalid_iteration_controls(maxiter: int) -> None:
    """The new diagnostic door rejects ambiguous budgets before entering JAX."""
    with pytest.raises(ValueError, match="maxiter must be a positive integer"):
        pgd_simplex_affine_result(jnp.eye(2), jnp.zeros(2), maxiter=maxiter)


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(tol=0.0)
@example(tol=-1.0)
@example(tol=float("nan"))
@example(tol=float("inf"))
@given(
    tol=st.one_of(
        st.floats(max_value=0.0, allow_nan=False),
        st.just(float("nan")),
        st.just(float("inf")),
    )
)
def test_affine_result_rejects_invalid_tolerance(tol: float) -> None:
    """A nonfinite or nonpositive threshold cannot certify stationarity."""
    with pytest.raises(ValueError, match="tol must be finite and strictly positive"):
        pgd_simplex_affine_result(jnp.eye(2), jnp.zeros(2), tol=tol)
