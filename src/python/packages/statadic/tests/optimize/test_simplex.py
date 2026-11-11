"""Projected-gradient descent over the simplex, against closed-form optima.

Every assertion is either a *feasibility* check (the iterate stays on the
simplex) or a comparison against an optimum with a closed form — a diagonal
covariance for the minimiser, a diagonal ``Q`` for the maximiser. That split is
deliberate: feasibility catches a projection bug, and the closed forms catch a
solver that is feasible and simply not converging.

The jit-vs-eager parity of the same solver is in
``tests/migration/test_parity_t05_8.py``; the analytic QP instances that replaced
the retired jaxopt BoxOSQP path are in
``tests/migration/test_kernels_migration.py``.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.optimize.simplex import (
    batched_nonnegative_quadratic_program,
    pgd_maximize_quadratic_form,
    pgd_minimize_quadratic_form,
)
from scipy.optimize import nnls

from .._budgets import examples
from .._strategies import prng_keys
from .._typing import ill_typed

if TYPE_CHECKING:
    from jax import Array


@pytest.mark.unit
@pytest.mark.reference
def test_one_nonnegative_qp_sweep_matches_independent_cyclic_minimizers() -> None:
    """A symmetric coupled quadratic has independently computable scalar updates."""
    quadratic = np.asarray([[4.0, 1.0, -0.5], [1.0, 3.0, 0.2], [-0.5, 0.2, 2.0]])
    linear = np.asarray([-2.0, -1.0, -0.4])
    expected = np.zeros(3)
    for index in range(3):
        other = sum(quadratic[index, j] * expected[j] for j in range(3) if j != index)
        expected[index] = max(0.0, -(linear[index] + other) / quadratic[index, index])
    result = batched_nonnegative_quadratic_program(
        jnp.asarray(quadratic), jnp.asarray(linear), max_sweeps=1
    )
    np.testing.assert_allclose(result.primal, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_batched_nonnegative_qp_matches_scipy_nnls() -> None:
    """Random SPD objectives, solutions, and active sets match SciPy NNLS."""
    rng = np.random.default_rng(564)
    dimension = 6
    n_problems = 24
    active_tolerance = 1e-10
    raw_design = rng.standard_normal((dimension, dimension))
    quadratic = raw_design.T @ raw_design + 0.5 * np.eye(dimension)
    design = np.linalg.cholesky(quadratic).T
    targets = rng.standard_normal((n_problems, dimension))
    linear = -(targets @ quadratic)

    solution = batched_nonnegative_quadratic_program(
        jnp.asarray(quadratic),
        jnp.asarray(linear),
        active_tolerance=active_tolerance,
    )
    reference = np.stack([nnls(design, design @ target)[0] for target in targets])
    reference_objective = np.einsum(
        "bi,ij,bj->b",
        0.5 * reference,
        quadratic,
        reference,
    ) + np.einsum("bi,bi->b", linear, reference)

    np.testing.assert_allclose(solution.primal, reference, atol=5e-8, rtol=5e-8)
    np.testing.assert_allclose(
        solution.objective,
        reference_objective,
        atol=5e-10,
        rtol=5e-10,
    )
    np.testing.assert_array_equal(solution.active_mask, reference <= active_tolerance)
    assert np.all(solution.converged)
    assert np.all(solution.input_finite)
    assert np.all(solution.convex)
    assert np.all(solution.bounded)


@pytest.mark.component
@pytest.mark.contract
def test_batched_nonnegative_qp_distinguishes_flat_failures() -> None:
    """Flat valid, unbounded-like, nonfinite, and nonconvex lanes are distinct."""
    flat_quadratic = jnp.diag(jnp.array([2.0, 0.0]))
    flat_valid = batched_nonnegative_quadratic_program(
        flat_quadratic,
        jnp.array([-4.0, 0.0]),
    )
    np.testing.assert_allclose(flat_valid.primal, [2.0, 0.0])
    np.testing.assert_array_equal(flat_valid.flat_mask, [False, True])
    assert bool(flat_valid.input_finite)
    assert bool(flat_valid.convex)
    assert bool(flat_valid.bounded)
    assert bool(flat_valid.converged)

    unbounded_like = batched_nonnegative_quadratic_program(
        flat_quadratic,
        jnp.array([-4.0, -1.0]),
    )
    np.testing.assert_array_equal(unbounded_like.primal, [0.0, 0.0])
    assert bool(unbounded_like.input_finite)
    assert bool(unbounded_like.convex)
    assert not bool(unbounded_like.bounded)
    assert not bool(unbounded_like.converged)
    assert np.isnan(float(unbounded_like.objective))
    assert np.isinf(float(unbounded_like.projected_gradient_norm))

    nonfinite = batched_nonnegative_quadratic_program(
        jnp.array([[1.0, jnp.nan], [jnp.nan, 1.0]]),
        jnp.array([-1.0, -1.0]),
    )
    np.testing.assert_array_equal(nonfinite.primal, [0.0, 0.0])
    assert not bool(nonfinite.input_finite)
    assert not bool(nonfinite.convex)
    assert not bool(nonfinite.bounded)
    assert not bool(nonfinite.converged)

    nonconvex = batched_nonnegative_quadratic_program(
        jnp.diag(jnp.array([1.0, -1.0])),
        jnp.array([-1.0, -1.0]),
    )
    np.testing.assert_array_equal(nonconvex.primal, [0.0, 0.0])
    assert bool(nonconvex.input_finite)
    assert not bool(nonconvex.convex)
    assert not bool(nonconvex.bounded)
    assert not bool(nonconvex.converged)


@pytest.mark.component
@pytest.mark.numerical
def test_batched_nonnegative_qp_singular_ill_conditioned_and_inactive() -> None:
    """Bounded singular, ill-conditioned, and all-inactive optima converge."""
    singular = batched_nonnegative_quadratic_program(
        jnp.ones((2, 2)),
        jnp.array([-2.0, -2.0]),
    )
    assert bool(singular.converged)
    assert bool(singular.bounded)
    assert float(jnp.sum(singular.primal)) == pytest.approx(2.0)
    assert float(singular.objective) == pytest.approx(-2.0)

    ill_conditioned = batched_nonnegative_quadratic_program(
        jnp.diag(jnp.array([1e-8, 1.0])),
        jnp.array([-1e-8, -1.0]),
    )
    np.testing.assert_allclose(ill_conditioned.primal, [1.0, 1.0])
    assert bool(ill_conditioned.converged)

    inactive = batched_nonnegative_quadratic_program(
        jnp.eye(2),
        jnp.array([1.0, 2.0]),
    )
    np.testing.assert_array_equal(inactive.primal, [0.0, 0.0])
    np.testing.assert_array_equal(inactive.active_mask, [True, True])
    assert int(inactive.sweeps) == 0
    assert bool(inactive.converged)


@pytest.mark.component
@pytest.mark.contract
def test_batched_nonnegative_qp_active_threshold_is_explicit() -> None:
    """The binding mask uses the configured absolute primal threshold."""
    solution = batched_nonnegative_quadratic_program(
        jnp.eye(2),
        jnp.array([-5e-11, -2e-10]),
        tolerance=1e-12,
        active_tolerance=1e-10,
    )
    np.testing.assert_allclose(solution.primal, [5e-11, 2e-10], atol=1e-16)
    np.testing.assert_array_equal(solution.active_mask, [True, False])


@pytest.mark.unit
@pytest.mark.contract
def test_batched_nonnegative_qp_is_transformable_without_callbacks() -> None:
    """The batched result survives make_jaxpr, nested jit, and an outer vmap."""
    quadratic = jnp.array([[2.0, 0.25], [0.25, 1.0]])
    linear = jnp.array([[-2.0, -1.0], [1.0, 2.0]])

    jaxpr = str(
        jax.make_jaxpr(batched_nonnegative_quadratic_program)(quadratic, linear)
    )
    assert "callback" not in jaxpr

    compiled = jax.jit(batched_nonnegative_quadratic_program)(quadratic, linear)
    compiled.primal.block_until_ready()
    assert np.all(compiled.converged)

    outer = jax.vmap(batched_nonnegative_quadratic_program)(
        jnp.stack([quadratic, quadratic]),
        jnp.stack([linear, linear]),
    )
    assert outer.primal.shape == (2, 2, 2)
    assert outer.converged.shape == (2, 2)


@pytest.mark.component
@pytest.mark.contract
def test_batched_nonnegative_qp_rejects_zero_dimension() -> None:
    """The eager contract rejects empty QPs before tracing an empty reduction."""
    with pytest.raises(ValueError, match="nonempty square rank-2"):
        batched_nonnegative_quadratic_program(
            jnp.empty((0, 0)),
            jnp.empty((0,)),
        )


_NONFINITE = st.sampled_from([float("nan"), float("inf")])
# Each control with the documented complement of its valid range.
_INVALID_STATIC_CONTROLS = st.one_of(
    st.tuples(
        st.just("max_sweeps"),
        st.one_of(
            st.integers(max_value=0),
            st.floats().filter(lambda v: not v.is_integer()),
            st.just(value=True),
        ),
    ),
    st.tuples(
        st.just("tolerance"),
        st.one_of(st.floats(max_value=0.0, allow_nan=False), _NONFINITE),
    ),
    st.tuples(
        st.sampled_from(
            ["active_tolerance", "curvature_tolerance", "linear_tolerance"]
        ),
        _NONFINITE,
    ),
)


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(keyword_and_value=("max_sweeps", 0))
@example(keyword_and_value=("max_sweeps", 1.5))
@example(keyword_and_value=("max_sweeps", True))
@example(keyword_and_value=("tolerance", 0.0))
@example(keyword_and_value=("tolerance", float("nan")))
@example(keyword_and_value=("tolerance", float("inf")))
@example(keyword_and_value=("active_tolerance", float("nan")))
@example(keyword_and_value=("active_tolerance", float("inf")))
@example(keyword_and_value=("curvature_tolerance", float("nan")))
@example(keyword_and_value=("curvature_tolerance", float("inf")))
@example(keyword_and_value=("linear_tolerance", float("nan")))
@example(keyword_and_value=("linear_tolerance", float("inf")))
@given(keyword_and_value=_INVALID_STATIC_CONTROLS)
def test_batched_nonnegative_qp_rejects_invalid_static_configuration(
    keyword_and_value: tuple[str, object],
) -> None:
    """Static loop and tolerance controls are finite positive scalars."""
    keyword, value = keyword_and_value
    with pytest.raises(
        (TypeError, ValueError),
        match=r"positive integer|finite and strictly positive|Static\[int\]",
    ):
        batched_nonnegative_quadratic_program(
            jnp.eye(2),
            jnp.zeros(2),
            **{keyword: ill_typed(value)},
        )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_batched_nonnegative_qp_preserves_float_precision(dtype: jnp.dtype) -> None:
    """Both supported JAX precisions converge without an implicit dtype cast."""
    quadratic = jnp.asarray([[2.0, 0.25], [0.25, 1.0]], dtype=dtype)
    linear = jnp.asarray([[-2.0, -1.0], [1.0, 2.0]], dtype=dtype)
    solution = batched_nonnegative_quadratic_program(
        quadratic,
        linear,
        tolerance=1e-6 if dtype == jnp.float32 else 1e-10,
    )
    assert solution.primal.dtype == dtype
    assert np.all(solution.converged)


@pytest.mark.unit
@pytest.mark.property
@examples(10)
@example(prng_key=jax.random.PRNGKey(42))
@given(prng_key=prng_keys())
def test_pgd_minimize_quadratic_form_simplex(prng_key: jax.Array) -> None:
    """The minimiser's iterate sums to one and stays non-negative."""
    n = 5
    a = jax.random.normal(prng_key, (n, n))
    sigma = a @ a.T + 0.5 * jnp.eye(n)
    w = pgd_minimize_quadratic_form(sigma, n_steps=2000)
    simplex_sum_tolerance = 1e-5
    weight_floor = -1e-6
    assert abs(float(jnp.sum(w)) - 1.0) < simplex_sum_tolerance
    assert float(jnp.min(w)) >= weight_floor


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.slow
def test_pgd_minimize_quadratic_form_closed_form() -> None:
    """Compare PGD with the diagonal covariance closed form.

    We use a diagonal sigma so the min-variance weights are inversely
    proportional to the diagonal (i.e., w_i ∝ 1/sigma_ii).
    """
    # Diagonal sigma: Sigma = diag([1, 4, 9])
    # min w'Σw s.t. sum(w)=1, w>=0 → w ∝ 1/[1,4,9]
    sigma = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
    w = pgd_minimize_quadratic_form(sigma, n_steps=10_000)

    # Closed-form long-only min-variance on diagonal: w_i = (1/sigma_i) / sum(1/sigma_j)
    inv = jnp.array([1.0, 0.25, 1.0 / 9.0])
    w_star = inv / jnp.sum(inv)

    assert jnp.allclose(w, w_star, atol=1e-3), (
        f"pgd_minimize_quadratic_form: {w}, expected: {w_star}"
    )


@pytest.mark.component
@pytest.mark.property
def test_pgd_maximize_quadratic_form_simplex() -> None:
    """The maximiser's iterate sums to one and stays non-negative."""
    q = jnp.array([[1.0, 0.5], [0.5, 2.0]])
    w = pgd_maximize_quadratic_form(q, n_steps=2000)
    simplex_sum_tolerance = 1e-5
    weight_floor = -1e-6
    assert abs(float(jnp.sum(w)) - 1.0) < simplex_sum_tolerance
    assert float(jnp.min(w)) >= weight_floor


@pytest.mark.component
@pytest.mark.numerical
def test_pgd_maximize_quadratic_form_concentrates_on_largest() -> None:
    """For diagonal Q = diag([1, 10]), the maximiser puts weight on asset 1."""
    q = jnp.diag(jnp.array([1.0, 10.0]))
    w = pgd_maximize_quadratic_form(q, n_steps=5000)
    # Maximum of w'Qw over simplex is at corner w=(0,1)
    concentration_threshold = 0.9
    assert w[1] > concentration_threshold, f"Expected weight on asset 1, got {w}"


def _nonnegative_active_set_oracle(
    quadratic: np.ndarray, linear: np.ndarray
) -> tuple[np.ndarray, float]:
    """Enumerate every free face of a tiny SPD nonnegative quadratic program."""
    candidates: list[tuple[np.ndarray, float]] = []
    tolerance = 1e-10
    for mask in product((False, True), repeat=len(linear)):
        free = np.asarray(mask)
        value = np.zeros_like(linear)
        value[free] = np.linalg.solve(quadratic[np.ix_(free, free)], -linear[free])
        gradient = quadratic @ value + linear
        if np.all(value >= -tolerance) and np.all(gradient[~free] >= -tolerance):
            objective = float(0.5 * value @ quadratic @ value + linear @ value)
            candidates.append((value, objective))
    assert candidates, "SPD finite QP must have an admissible active-set solution"
    return min(candidates, key=lambda candidate: candidate[1])


@pytest.mark.unit
@pytest.mark.reference
@examples(12)
@given(
    entries=st.lists(st.integers(-3, 3), min_size=9, max_size=9),
    costs=st.lists(st.integers(-4, 4), min_size=3, max_size=3),
)
def test_nonnegative_qp_matches_tiny_active_sets_and_independent_kkt(
    entries: list[int], costs: list[int]
) -> None:
    """Generated SPD faces test objective, active coordinates and actual KKT error."""
    factor = np.asarray(entries, dtype=np.float64).reshape(3, 3)
    quadratic = np.eye(3) + 0.1 * factor.T @ factor
    linear = np.asarray(costs, dtype=np.float64)
    expected, expected_objective = _nonnegative_active_set_oracle(quadratic, linear)
    active_tolerance = 1e-10
    result = jax.jit(
        batched_nonnegative_quadratic_program, static_argnames=("active_tolerance",)
    )(quadratic, linear, active_tolerance=active_tolerance)
    actual = np.asarray(result.primal)
    gradient = quadratic @ actual + linear
    projected = np.max(np.abs(actual - np.maximum(actual - gradient, 0.0)))
    tolerance = 5e-8

    assert bool(result.converged)
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert float(result.objective) == pytest.approx(expected_objective, abs=1e-8)
    np.testing.assert_array_equal(result.active_mask, expected <= active_tolerance)
    assert projected < tolerance
    assert float(result.projected_gradient_norm) == pytest.approx(projected, abs=1e-14)
    np.testing.assert_allclose(actual * gradient, 0.0, atol=tolerance)


@pytest.mark.component
@pytest.mark.numerical
def test_nonnegative_qp_interior_derivatives_match_the_linear_solution() -> None:
    """Away from binding-set changes, both AD modes recover minus the inverse."""
    quadratic = np.asarray([[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.0]])
    linear = -(quadratic @ np.asarray([1.0, 2.0, 1.5]))

    def solve(cost: jax.Array) -> jax.Array:
        return batched_nonnegative_quadratic_program(
            jnp.asarray(quadratic), cost
        ).primal

    expected = -np.linalg.solve(quadratic, np.eye(3))
    forward = jax.jit(jax.jacfwd(solve))(jnp.asarray(linear))
    reverse = jax.jit(jax.jacrev(solve))(jnp.asarray(linear))
    np.testing.assert_allclose(forward, expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(reverse, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_quadratic_maximization_reaches_psd_vertex_bound_and_retains_concave_interior(
    dtype: type[np.float32 | np.float64],
) -> None:
    """Uniform-start stationary minima cannot masquerade as PSD global maxima."""
    factor = np.asarray([[1.0, 0.3, -0.7], [0.2, 1.1, 0.4], [0.0, -0.1, 0.8]])
    matrices = np.stack(
        (np.eye(3), np.diag([2.0, 2.0, 0.5]), factor.T @ factor, -np.eye(3))
    ).astype(dtype)

    def evaluate_pgd_maximize_quadratic_form(matrix: jax.Array) -> Array:
        return pgd_maximize_quadratic_form(matrix, 32)

    result = jax.jit(jax.vmap(evaluate_pgd_maximize_quadratic_form))(
        jnp.asarray(matrices)
    )
    margin = 32.0 * float(np.finfo(dtype).eps)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=margin)
    assert np.all(np.asarray(result) >= 0.0)
    for matrix, value in zip(matrices[:3], result[:3], strict=True):
        assert float(value @ matrix @ value) == pytest.approx(
            float(np.diag(matrix).max()), abs=margin
        )
    np.testing.assert_array_equal(result[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result[3], np.full(3, 1 / 3), atol=margin)
    assert result.dtype == dtype


@pytest.mark.component
@pytest.mark.property
@examples(12)
@given(entries=st.lists(st.integers(-4, 4), min_size=9, max_size=9))
def test_generated_psd_quadratic_maximization_attains_the_global_vertex_value(
    entries: list[int],
) -> None:
    """PSD Gram inputs include rank-deficient, tied and identically flat cases."""
    factor = np.asarray(entries, dtype=np.float64).reshape(3, 3) / 4.0
    quadratic = factor.T @ factor
    value = pgd_maximize_quadratic_form(jnp.asarray(quadratic), n_steps=32)
    assert float(value.sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.asarray(value) >= 0.0)
    assert float(value @ quadratic @ value) == pytest.approx(
        float(np.diag(quadratic).max()), abs=1e-12
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("maximize", [False, True], ids=["minimize", "maximize"])
def test_pure_quadratic_float32_preserves_dtype_with_global_x64_enabled(
    *,
    maximize: bool,
) -> None:
    """Optax integer-division promotion cannot change the loop or result dtype."""

    def evaluate_solve(value: jax.Array) -> Array:
        return solve(value, n_steps=64)

    with jax.enable_x64(new_val=True):
        matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32))
        solve = pgd_maximize_quadratic_form if maximize else pgd_minimize_quadratic_form
        eager = solve(matrix, n_steps=64)

        compiled = jax.jit(evaluate_solve)(matrix)
        assert eager.dtype == jnp.float32
        assert compiled.dtype == jnp.float32
        np.testing.assert_allclose(eager, compiled, atol=1e-7)
        assert jax.config.x64_enabled


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("scale", [-1e30, 1.8e38])
def test_quadratic_maximization_ignores_nonfinite_ascent_candidates(
    scale: float,
) -> None:
    """Finite vertex/uniform witnesses survive overflowing ascent intermediates."""
    matrix = jnp.eye(3, dtype=jnp.float32) * scale

    def evaluate_pgd_maximize_quadratic_form(a: jax.Array) -> Array:
        return pgd_maximize_quadratic_form(a, n_steps=32)

    result = jax.jit(evaluate_pgd_maximize_quadratic_form)(matrix)
    assert np.isfinite(result).all()
    assert result.dtype == jnp.float32
    expected = np.full(3, 1 / 3) if scale < 0 else np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
