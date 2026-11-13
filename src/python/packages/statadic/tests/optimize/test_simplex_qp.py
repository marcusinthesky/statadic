"""Independent objective, domain and KKT checks for capped-simplex QPs."""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jaxtyping import TypeCheckError
from scipy.optimize import minimize
from statadic.core.results import SolverStatus
from statadic.optimize.simplex import SimplexQPSolution, simplex_qp_result

from .._budgets import examples

#: Nonpositive or nonfinite: the documented complement of a valid tolerance.
_INVALID_TOLERANCES = st.one_of(
    st.floats(max_value=0.0, allow_nan=False),
    st.just(float("nan")),
    st.just(float("inf")),
)


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize(
    ("maximum_weight", "expected"),
    [(1.0, [6 / 11, 3 / 11, 2 / 11]), (0.5, [0.5, 0.3, 0.2])],
)
def test_capped_qp_matches_analytic_diagonal_optimum(
    maximum_weight: float, expected: list[float]
) -> None:
    """Inverse curvatures give an interior solution or the correct active cap."""
    tolerance = 1e-10
    gap_bound = 1e-8
    result = simplex_qp_result(
        jnp.diag(jnp.array([1.0, 2.0, 3.0])),
        jnp.zeros(3),
        maximum_weight=maximum_weight,
        tolerance=tolerance,
    )
    np.testing.assert_allclose(result.solver.value, expected, atol=1e-8, rtol=0)
    assert isinstance(result, SimplexQPSolution)
    assert bool(result.solver.converged)
    assert bool(result.backend_converged)
    assert bool(result.feasible)
    assert float(result.solver.residual_norm) <= tolerance
    assert float(result.optimality_gap) < gap_bound
    assert int(result.solver.status) == SolverStatus.SUCCESS


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("maximum_weight", [1.0, 0.3])
def test_capped_qp_agrees_with_independent_scipy(maximum_weight: float) -> None:
    """An unrelated constrained solver checks a coupled quadratic and linear term."""
    generator = np.random.default_rng(20261025)
    design = generator.normal(size=(8, 5))
    quadratic = design.T @ design
    linear = np.array([0.2, -0.7, 0.4, 1.2, -0.8])

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * weights @ quadratic @ weights + linear @ weights)

    reference = minimize(
        objective,
        np.full(5, 0.2),
        jac=lambda weights: quadratic @ weights + linear,
        bounds=[(0.0, maximum_weight)] * 5,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert reference.success
    result = simplex_qp_result(
        jnp.asarray(quadratic),
        jnp.asarray(linear),
        maximum_weight=maximum_weight,
        tolerance=1e-9,
    )
    assert bool(result.solver.converged)
    np.testing.assert_allclose(result.solver.value, reference.x, atol=2e-7, rtol=0)
    assert float(result.objective) == pytest.approx(reference.fun, abs=2e-9)
    assert (
        float(result.objective) - reference.fun <= float(result.optimality_gap) + 1e-10
    )


@pytest.mark.component
@pytest.mark.numerical
def test_capped_qp_accepts_ambient_indefinite_tangent_convex_objective() -> None:
    """A constant-on-simplex negative curvature must not reject a valid problem."""
    quadratic = jnp.diag(jnp.array([1.0, 2.0, 3.0])) - 10.0 * jnp.ones((3, 3))
    assert float(jnp.min(jnp.linalg.eigvalsh(quadratic))) < 0.0
    result = simplex_qp_result(quadratic, jnp.zeros(3), tolerance=1e-9)
    assert bool(result.convex)
    assert bool(result.solver.converged)
    np.testing.assert_allclose(result.solver.value, [6 / 11, 3 / 11, 2 / 11], atol=1e-7)
    assert float(result.objective) == pytest.approx(3 / 11 - 5.0, abs=1e-9)


@pytest.mark.component
@pytest.mark.regression
def test_large_constant_curvature_cannot_hide_tangent_nonconvexity() -> None:
    """An irrelevant constant-on-simplex offset must not loosen the PSD gate."""
    quadratic = 1e15 * jnp.ones((3, 3)) - jnp.eye(3)
    linear = -(quadratic @ jnp.full(3, 1 / 3))
    result = simplex_qp_result(quadratic, linear)
    assert not bool(result.convex)
    assert not bool(result.solver.converged)
    assert int(result.solver.status) == SolverStatus.FAILED


@pytest.mark.component
@pytest.mark.regression
def test_large_constant_curvature_cannot_hide_material_asymmetry() -> None:
    """An offset cannot authorize the wrong gradient of an asymmetric matrix."""
    antisymmetric = jnp.array([[0.0, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    quadratic = 1e15 * jnp.ones((3, 3)) + jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    quadratic += antisymmetric
    linear = -(quadratic @ jnp.full(3, 1 / 3))
    result = simplex_qp_result(quadratic, linear)
    assert not bool(result.convex)
    assert not bool(result.solver.converged)
    assert int(result.solver.status) == SolverStatus.FAILED


@pytest.mark.component
@pytest.mark.numerical
def test_capped_qp_handles_a_linear_program_and_singular_quadratic() -> None:
    """Semidefinite objectives remain valid without adding a ridge."""
    linear_program = simplex_qp_result(
        jnp.zeros((3, 3)),
        jnp.array([0.0, 1.0, 2.0]),
        maximum_weight=0.6,
        tolerance=1e-9,
    )
    assert bool(linear_program.solver.converged)
    np.testing.assert_allclose(linear_program.solver.value, [0.6, 0.4, 0.0], atol=2e-9)
    singular = simplex_qp_result(
        jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
        jnp.zeros(3),
        tolerance=1e-9,
    )
    assert bool(singular.solver.converged)
    # The first two weights are not identified; their sum and the objective are.
    assert float(jnp.sum(singular.solver.value[:2])) == pytest.approx(2 / 3, abs=1e-8)
    assert float(singular.solver.value[2]) == pytest.approx(1 / 3, abs=1e-8)
    assert float(singular.objective) == pytest.approx(1 / 3, abs=1e-9)


@pytest.mark.component
@pytest.mark.numerical
def test_capped_qp_handles_flat_fixed_uniform_and_singleton_domains() -> None:
    """Direct witnesses avoid singular or zero-interior backend problems."""
    flat = simplex_qp_result(jnp.zeros((4, 4)), jnp.ones(4))
    fixed = simplex_qp_result(
        jnp.eye(4), jnp.array([1.0, -2.0, 3.0, -4.0]), maximum_weight=0.25
    )
    singleton = simplex_qp_result(jnp.array([[-3.0]]), jnp.array([2.0]))
    for result, expected in [
        (flat, [0.25] * 4),
        (fixed, [0.25] * 4),
        (singleton, [1.0]),
    ]:
        np.testing.assert_array_equal(result.solver.value, expected)
        assert bool(result.solver.converged)
        assert bool(result.convex)
        assert int(result.solver.iterations) == 0
        np.testing.assert_array_equal(result.optimality_gap, 0.0)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("sites", [[3.0, 9.0], [3.0, 3.0, 9.0], [1.0, 3.0, 9.0]])
def test_capped_qp_certifies_weakly_active_dirac_vertex(sites: list[float]) -> None:
    """A matching Dirac candidate has zero energy and zero active multipliers."""
    locations = np.asarray(sites)
    gram = 2 * np.abs(locations[:, None] - locations[None, :])
    cross = 2 * np.abs(locations - 3.0)
    tolerance = 1e-9
    result = simplex_qp_result(
        jnp.asarray(-2 * gram),
        jnp.asarray(2 * cross),
        tolerance=tolerance,
        max_steps=1,
    )
    value = np.asarray(result.solver.value)
    gradient = -2 * gram @ value + 2 * cross
    expected = np.zeros(len(sites))
    expected[sites.index(3.0)] = 1.0
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(2 * cross @ value - value @ gram @ value, 0.0)
    np.testing.assert_array_equal(gradient, np.zeros(len(sites)))
    assert bool(result.solver.converged)
    assert bool(result.feasible)
    assert int(result.solver.iterations) == 0
    assert float(result.solver.residual_norm) <= tolerance
    np.testing.assert_array_equal(result.optimality_gap, 0.0)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize(
    ("cap", "expected", "objective"),
    [(1.0, [0.75, 0.25], 0.75), (0.6, [0.6, 0.4], 0.84)],
)
def test_capped_qp_rejects_nonoptimal_or_infeasible_vertex_proposal(
    cap: float, expected: list[float], objective: float
) -> None:
    """The independent interval quadratic has an interior or capped optimum."""
    # With t=w[0], this is 4*(t-3/4)^2+3/4 on [1-cap, cap].
    result = simplex_qp_result(
        jnp.array([[0.0, -4.0], [-4.0, 0.0]]),
        jnp.array([1.0, 3.0]),
        maximum_weight=cap,
        tolerance=1e-9,
    )
    assert bool(result.solver.converged)
    assert int(result.solver.iterations) > 0
    np.testing.assert_allclose(result.solver.value, expected, rtol=0, atol=1e-8)
    assert float(result.objective) == pytest.approx(objective, abs=1e-9)


@pytest.mark.component
@pytest.mark.contract
def test_capped_qp_preserves_mixed_statuses_under_jit_and_vmap() -> None:
    """Successful, exhausted, nonconvex and nonfinite lanes keep their evidence."""
    quadratics = jnp.stack(
        [
            jnp.eye(3),
            jnp.diag(jnp.array([1.0, 2.0, 3.0])),
            -jnp.eye(3),
            jnp.full((3, 3), jnp.nan),
        ]
    )

    def evaluate_simplex_qp_result(matrix: jax.Array) -> SimplexQPSolution:
        return simplex_qp_result(matrix, jnp.zeros(3), max_steps=1)

    result = jax.jit(jax.vmap(evaluate_simplex_qp_result))(quadratics)
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
    tolerance = 1e-7
    assert float(result.solver.residual_norm[1]) > tolerance
    assert not bool(result.backend_converged[1])
    assert not bool(result.convex[2])
    assert not bool(result.input_finite[3])
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(result))


@pytest.mark.component
@pytest.mark.contract
def test_capped_qp_treats_invalid_dynamic_bounds_as_failed_lanes() -> None:
    """A traced cap cannot cause silent renormalization or a host exception."""

    def evaluate_simplex_qp_result(cap: jax.Array) -> SimplexQPSolution:
        return simplex_qp_result(jnp.eye(3), jnp.zeros(3), maximum_weight=cap)

    result = jax.jit(jax.vmap(evaluate_simplex_qp_result))(
        jnp.array([1.0, 1 / 3, 0.2, -1.0, jnp.nan])
    )
    np.testing.assert_array_equal(
        result.solver.converged, [True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        result.bounds_valid, [True, True, False, False, False]
    )
    assert not bool(result.feasible[2])
    material_violation = 0.1
    assert float(result.primal_residual[2]) > material_violation
    assert np.isnan(float(result.optimality_gap[2]))


@pytest.mark.component
@pytest.mark.contract
def test_capped_qp_rejects_nonsymmetric_input_without_a_false_success() -> None:
    """QPAX's internal symmetrization must not conceal an invalid public input."""
    result = simplex_qp_result(jnp.array([[1.0, 0.5], [0.0, 1.0]]), jnp.zeros(2))
    assert bool(result.finite)
    assert not bool(result.convex)
    assert not bool(result.solver.converged)
    assert int(result.solver.status) == SolverStatus.FAILED


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_capped_qp_respects_input_precision_and_requested_tolerance(
    dtype: jnp.dtype,
) -> None:
    """A representable float32 problem succeeds without changing global precision."""
    previous = jax.config.x64_enabled
    tolerance = 1e-7
    result = simplex_qp_result(
        jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=dtype)),
        jnp.zeros(3, dtype=dtype),
        maximum_weight=0.5,
    )
    assert result.solver.value.dtype == dtype
    assert result.objective.dtype == dtype
    assert bool(result.solver.converged)
    assert float(result.solver.residual_norm) <= tolerance
    np.testing.assert_allclose(result.solver.value, [0.5, 0.3, 0.2], atol=3e-7, rtol=0)
    assert jax.config.x64_enabled == previous


@pytest.mark.component
@pytest.mark.contract
def test_capped_qp_does_not_floor_an_unattained_float32_tolerance() -> None:
    """An overly strict request stays failed instead of silently claiming success."""
    tolerance = 1e-12
    max_steps = 30
    result = simplex_qp_result(
        jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)),
        jnp.zeros(3, dtype=jnp.float32),
        maximum_weight=0.5,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    assert not bool(result.solver.converged)
    assert not bool(result.backend_converged)
    assert int(result.solver.status) == SolverStatus.MAX_STEPS
    assert int(result.solver.iterations) == max_steps


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(max_steps=0)
@example(max_steps=-1)
@example(max_steps=True)
@given(max_steps=st.one_of(st.integers(max_value=0), st.just(value=True)))
def test_capped_qp_rejects_invalid_iteration_controls(max_steps: int) -> None:
    """Iteration budgets are unambiguous positive static integers."""
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        simplex_qp_result(jnp.eye(2), jnp.zeros(2), max_steps=max_steps)


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(tolerance=0.0)
@example(tolerance=-1.0)
@example(tolerance=np.nan)
@example(tolerance=np.inf)
@given(tolerance=_INVALID_TOLERANCES)
def test_capped_qp_rejects_invalid_tolerances(tolerance: float) -> None:
    """Nonpositive or nonfinite tolerances cannot authorize convergence."""
    with pytest.raises(
        ValueError, match="tolerance must be finite and strictly positive"
    ):
        simplex_qp_result(jnp.eye(2), jnp.zeros(2), tolerance=tolerance)


@pytest.mark.component
@pytest.mark.contract
def test_capped_qp_rejects_invalid_shapes() -> None:
    """Static array misuse fails before the backend is traced."""
    with pytest.raises(ValueError, match="nonempty square rank-2"):
        simplex_qp_result(jnp.zeros((0, 0)), jnp.zeros(0))
    with pytest.raises(TypeCheckError, match="parameter 'linear'"):
        simplex_qp_result(jnp.eye(3), jnp.zeros(2))


def _capped_face_oracle(
    quadratic: np.ndarray, linear: np.ndarray, cap: float
) -> tuple[np.ndarray, float]:
    """Solve each tiny box face with a separate equality-constrained KKT system."""
    candidates: list[tuple[np.ndarray, float]] = []
    free_state, upper_state = 1, 2
    tolerance = 1e-10
    for choices in product((0, free_state, upper_state), repeat=len(linear)):
        state = np.asarray(choices)
        free = state == free_state
        value = cap * (state == upper_state).astype(np.float64)
        if np.any(free):
            size = int(free.sum())
            kkt = np.block(
                [
                    [quadratic[np.ix_(free, free)], np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.r_[-linear[free] - (quadratic @ value)[free], 1.0 - value.sum()]
            value[free] = np.linalg.solve(kkt, rhs)[:-1]
        if (
            abs(value.sum() - 1.0) <= tolerance
            and np.all(value >= -tolerance)
            and np.all(value <= cap + tolerance)
        ):
            objective = float(0.5 * value @ quadratic @ value + linear @ value)
            candidates.append((value, objective))
    assert candidates, "A nonempty capped simplex must have an optimum on a face"
    return min(candidates, key=lambda candidate: candidate[1])


def _capped_vertices(dimension: int, cap: float) -> np.ndarray:
    """Enumerate vertices independently of the solver's sorted-gradient shortcut."""
    vertices: list[np.ndarray] = []
    tolerance = 1e-12
    for free in range(dimension):
        fixed = np.delete(np.arange(dimension), free)
        for bounds in product((0.0, cap), repeat=dimension - 1):
            vertex = np.zeros(dimension)
            vertex[fixed] = bounds
            vertex[free] = 1.0 - vertex.sum()
            if -tolerance <= vertex[free] <= cap + tolerance:
                vertices.append(vertex)
    return np.stack(vertices)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("cap", [0.4, 0.6, 1.0])
@examples(8)
@given(
    diagonal=st.lists(st.integers(1, 8), min_size=3, max_size=3),
    costs=st.lists(st.integers(-2, 2), min_size=3, max_size=3),
)
def test_generated_tangent_convex_qp_matches_faces_and_vertex_gap(
    cap: float, diagonal: list[int], costs: list[int]
) -> None:
    """Negative ambient curvature cannot suppress a valid tangent-convex solution."""
    quadratic = np.diag(np.asarray(diagonal) / 4.0 + 0.25) - 9.0 * np.ones((3, 3))
    linear = np.asarray(costs, dtype=np.float64) / 10.0
    expected, optimum = _capped_face_oracle(quadratic, linear, cap)
    result = simplex_qp_result(
        jnp.asarray(quadratic),
        jnp.asarray(linear),
        maximum_weight=cap,
        tolerance=1e-9,
    )
    value = np.asarray(result.solver.value)
    gradient = quadratic @ value + linear
    gap = max(
        float(gradient @ value - np.min(_capped_vertices(3, cap) @ gradient)), 0.0
    )
    feasibility_tolerance = 1e-8
    gap_tolerance = 1e-7

    assert bool(result.convex)
    assert bool(result.feasible)
    assert abs(value.sum() - 1.0) < feasibility_tolerance
    assert np.all(value >= -feasibility_tolerance)
    assert np.all(value <= cap + feasibility_tolerance)
    np.testing.assert_allclose(value, expected, rtol=1e-7, atol=1e-7)
    assert float(result.objective) == pytest.approx(optimum, abs=1e-8)
    assert gap < gap_tolerance
    assert float(result.optimality_gap) == pytest.approx(gap, abs=1e-12)
    assert float(result.objective) - optimum <= gap + 1e-10
