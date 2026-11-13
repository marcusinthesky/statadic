# standalone package-maintenance entry point
"""Tests for :mod:`statadic.optimize.psd` — every PSD repair the module ships.

Restructured in t65.4, when the NumPy Higham/Dykstra backend
(``statadic.optimize._psd_numpy``) was deleted and the alternating-projection
kernel was replaced by Qi-Sun semismooth Newton. The former cross-port parity
suite is gone with the second port; in its place this file carries an
**independent oracle**: Dykstra alternating projections run to 5000 iterations,
implemented locally rather than imported, so the production solver is checked
against a different algorithm rather than against a copy of itself.

Four things are asserted, in rough order of what would hurt most if it broke:

1. **Nearest, not merely feasible.** The returned matrix agrees with the oracle.
   This is the claim the old NumPy path could not make: its 200-iteration budget
   never reached its own ``1e-10`` tolerance on production-shaped inputs and
   landed ``7e-4`` from the converged answer.
2. **Structural invariants** — unit diagonal, preserved marginal variances,
   PSD tight enough for ``cholesky``.
3. **Precision policy** — the nearest-matrix contract holds in float64; float32
   is supported and reports ``converged=False`` rather than overclaiming.
4. **Traceability** — every repair composes under ``jit`` and ``vmap`` with a
   fixed result structure, since these run inside stage kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from jax import lax
from statadic.optimize.psd import (
    _NEAREST_CLAIM_CEILING,
    DEFAULT_RIDGE_EPS,
    _positive_energy,
    _positive_projection,
    eig_clip,
    gerber_shrink_psd,
    nearest_correlation,
    nearest_covariance,
    repair_diagnostics,
    ridge_psd,
)

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import ill_typed

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from jax import Array

_INDEFINITE_TRIPLE = np.array([[1.0, 0.9, -0.3], [0.9, 1.0, 0.9], [-0.3, 0.9, 1.0]])


@pytest.mark.unit
@pytest.mark.numerical
def test_spectral_derivatives_at_repeated_eigenvalues() -> None:
    """Spectral divided differences remain valid at repeated eigenvalues."""
    matrix = jnp.diag(jnp.asarray([2.0, 2.0, -1.0]))
    direction = jnp.asarray([[1.0, 0.3, 0.6], [0.3, -2.0, 0.9], [0.6, 0.9, 4.0]])
    expected = np.asarray(direction) * np.asarray(
        [[1, 1, 2 / 3], [1, 1, 2 / 3], [2 / 3, 2 / 3, 0]]
    )
    gradient, action = jax.linearize(jax.grad(_positive_energy), matrix)
    np.testing.assert_allclose(gradient, np.diag([2, 2, 0]), atol=1e-12)
    np.testing.assert_allclose(action(direction), expected, atol=1e-12)
    _, pullback = jax.vjp(_positive_projection, matrix)
    np.testing.assert_allclose(pullback(direction)[0], expected, atol=1e-12)
    step = 1e-5
    finite_difference = (
        _positive_projection(matrix + step * direction)
        - _positive_projection(matrix - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(finite_difference, expected, atol=1e-9)


@pytest.mark.unit
@pytest.mark.numerical
def test_cone_vertex_uses_selected_zero_generalized_derivative() -> None:
    """At the nonsmooth vertex the registered selection is explicitly zero."""
    zero = jnp.zeros((2, 2))
    _, tangent = jax.jvp(_positive_projection, (zero,), (jnp.eye(2),))
    np.testing.assert_array_equal(tangent, zero)


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_eig_clip_has_exact_interior_derivatives_at_repeated_eigenvalues(
    sign: float,
) -> None:
    """Clipping is identity or constant on the two shifted definite interiors."""
    matrix = sign * jnp.eye(2)
    direction = jnp.asarray([[1.0, 0.3], [0.3, -2.0]])
    weights = jnp.asarray([[1.0, 2.0], [2.0, 4.0]])

    def repair(value: Array) -> Array:
        return eig_clip(value, eps=0.1)

    def objective(value: Array) -> Array:
        return jnp.sum(weights * repair(value))

    expected = matrix if sign > 0 else 0.1 * jnp.eye(2)
    derivative = weights if sign > 0 else jnp.zeros_like(weights)
    np.testing.assert_allclose(jax.jit(repair)(matrix), expected, atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(matrix), derivative, atol=1e-12
    )
    _, tangent = jax.jvp(repair, (matrix,), (direction,))
    np.testing.assert_allclose(tangent, direction if sign > 0 else 0, atol=1e-12)
    np.testing.assert_allclose(jax.hessian(objective)(matrix), 0, atol=1e-12)
    _, hvp = jax.jvp(jax.grad(objective), (matrix,), (direction,))
    np.testing.assert_allclose(hvp, 0, atol=1e-12)
    batched = jax.jit(jax.vmap(repair))(jnp.stack([matrix, 2 * matrix]))
    np.testing.assert_allclose(batched[0], expected, atol=1e-12)
    np.testing.assert_allclose(
        batched[1], 2 * matrix if sign > 0 else expected, atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.regression
def test_eig_clip_mixed_spectrum_has_repeated_eigenvalue_safe_first_derivative() -> (
    None
):
    """The clipped scalar function gives an independent divided-difference map."""
    matrix = jnp.diag(jnp.asarray([2.0, 2.0, -1.0]))
    direction = jnp.asarray([[1.0, 0.3, 0.6], [0.3, -2.0, 0.9], [0.6, 0.9, 4.0]])
    expected = np.asarray(direction) * np.asarray(
        [[1, 1, 1.9 / 3], [1, 1, 1.9 / 3], [1.9 / 3, 1.9 / 3, 0]]
    )

    def repair(value: Array) -> Array:
        return eig_clip(value, eps=0.1)

    _, tangent = jax.jvp(repair, (matrix,), (direction,))
    np.testing.assert_allclose(tangent, expected, atol=1e-12)
    _, pullback = jax.vjp(repair, matrix)
    np.testing.assert_allclose(pullback(direction)[0], expected, atol=1e-12)


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_eig_clip_keeps_finite_extreme_primal_and_mixed_derivative(dtype: str) -> None:
    """A finite floor must not overflow by shifting the original spectrum."""
    with jax.enable_x64(new_val=dtype == "float64"):
        high = float(np.finfo(dtype).max) * 0.75
        floor = high * 0.5
        matrix = jnp.diag(jnp.asarray([high, -high], dtype=dtype))
        direction = jnp.asarray([[2.0, 0.4], [0.4, -3.0]], dtype=dtype)

        def repair(value: Array) -> Array:
            return eig_clip(value, eps=floor)

        def pushforward(value: Array, tangent: Array) -> tuple[Array, Array]:
            return jax.jvp(repair, (value,), (tangent,))

        def total(value: Array) -> Array:
            return jnp.sum(repair(value))

        primal, tangent = jax.jit(pushforward)(matrix, direction)
        expected = np.diag(np.asarray([high, floor], dtype=dtype))
        expected_tangent = np.asarray([[2.0, 0.1], [0.1, 0.0]], dtype=dtype)
        assert np.isfinite(primal).all()
        assert np.isfinite(tangent).all()
        np.testing.assert_allclose(primal, expected, rtol=1e-6)
        np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-6, atol=1e-7)
        gradient = jax.jit(jax.grad(total))(matrix)
        np.testing.assert_allclose(
            gradient, [[1.0, 0.25], [0.25, 0.0]], rtol=1e-6, atol=1e-7
        )
        batch = jax.jit(jax.vmap(repair))(jnp.stack([matrix, -matrix]))
        np.testing.assert_allclose(batch[0], expected, rtol=1e-6)
        np.testing.assert_allclose(batch[1], expected[::-1, ::-1], rtol=1e-6)


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fraction", [0.25, 0.75])
@pytest.mark.parametrize("low", ["small-positive", "negative-high"])
def test_eig_clip_preserves_small_eigenvalues_and_absolute_floor(
    dtype: str, fraction: float, low: str
) -> None:
    """Normalizing the whole spectrum must not erase a resolved small value."""
    with jax.enable_x64(new_val=dtype == "float64"):
        high = float(np.finfo(dtype).max) * fraction
        below = low == "negative-high"
        matrix = jnp.diag(jnp.asarray([high, -high if below else 1.0], dtype=dtype))
        direction = jnp.asarray([[2.0, 0.4], [0.4, 3.0]], dtype=dtype)
        floor = 0.1

        def repair(value: Array) -> Array:
            return eig_clip(value, eps=floor)

        def differentiated(value: Array, tangent: Array) -> tuple[Array, Array]:
            return jax.jvp(repair, (value,), (tangent,))

        expected = np.diag(np.asarray([high, floor if below else 1.0], dtype=dtype))
        expected_tangent = (
            np.asarray([[2.0, 0.2], [0.2, 0.0]], dtype=dtype)
            if below
            else np.asarray(direction)
        )
        eager, tangent = differentiated(matrix, direction)
        compiled, compiled_tangent = jax.jit(differentiated)(matrix, direction)
        for actual in (eager, compiled):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        for actual in (tangent, compiled_tangent):
            np.testing.assert_allclose(actual, expected_tangent, rtol=1e-6, atol=1e-7)


@pytest.mark.unit
@pytest.mark.numerical
def test_spectral_floor_tangent_and_lax_composition() -> None:
    """A varying floor contributes on the clipped eigenspace, once per call."""
    matrix = jnp.diag(jnp.asarray([4.0, 1.0, -2.0]))
    direction = jnp.asarray([[1.0, 0.3, 0.6], [0.3, -2.0, 0.9], [0.6, 0.9, 4.0]])
    floor = jnp.asarray(2.0)
    expected = np.asarray(direction) * np.asarray(
        [[1, 2 / 3, 1 / 3], [2 / 3, 0, 0], [1 / 3, 0, 0]]
    ) + np.diag([0.0, 0.5, 0.5])
    _, tangent = jax.jvp(
        _positive_projection, (matrix, floor), (direction, jnp.asarray(0.5))
    )
    np.testing.assert_allclose(tangent, expected, atol=1e-12)
    step = 1e-5
    finite_difference = (
        _positive_projection(matrix + step * direction, floor + step * 0.5)
        - _positive_projection(matrix - step * direction, floor - step * 0.5)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite_difference, atol=1e-9)

    def scan(value: Array) -> Array:
        def step(carry: Array, offset: Array) -> tuple[Array, Array]:
            result = _positive_projection(value, floor + offset)
            return carry + result, result

        return lax.scan(step, jnp.zeros_like(value), jnp.asarray([0.0, 0.5]))[0]

    np.testing.assert_allclose(
        jax.jit(scan)(matrix), np.diag([8.0, 4.5, 4.5]), atol=1e-12
    )

    def trace_scan(value: Array) -> Array:
        return jnp.trace(scan(value))

    scan_gradient = jax.jit(jax.grad(trace_scan))(matrix)
    np.testing.assert_allclose(scan_gradient, np.diag([2.0, 0.0, 0.0]), atol=1e-12)


def _symmetric_matrices(n: int, bound: float) -> SearchStrategy[np.ndarray]:
    """Draw symmetric ``n``-by-``n`` matrices with entries within ``±bound``.

    Any symmetric matrix is a valid repair input: indefinite draws are the point.
    """
    return hnp.arrays(
        np.float64,
        (n, n),
        elements=st.floats(-bound, bound, allow_nan=False, allow_infinity=False),
    ).map(lambda a: 0.5 * (a + a.T))


#: The unified ridge default. Asserted literally so a change to the constant
#: has to be made here too, deliberately, rather than passing silently.
_EXPECTED_RIDGE_EPS = 1e-8
#: Slack for the post-repair minimum eigenvalue on a float32 default backend.
_MIN_EIGENVALUE_TOLERANCE = -1e-6
_CHOLESKY_EIGENVALUE_TOLERANCE = 1e-12
_TRACE_PRESERVATION_TOLERANCE = 1e-8
_PSD_CLASSIFICATION_TOLERANCE = 1e-10
_MONTE_CARLO_TRACE_TOLERANCE = 1e-7
#: Distance from the oracle that the float64 solver is required to reach *when
#: it reports ``converged``. The floor is the eigenvalue floor itself
#: (~``8·n·eps·λ_max``), not the solver.
_ORACLE_AGREEMENT_TOLERANCE = 1e-9
#: Distance from the oracle allowed on a run that reports ``converged=False``.
#: Retained only for :func:`test_a_too_tight_tolerance_costs_iterations_not_accuracy`,
#: which asks an unreachable tolerance for a deliberate non-convergence. The
#: former ``_STALL_RESIDUAL_CEILING`` that sat beside it is **deleted**: it
#: codified the ``~4e-08`` "boundary plateau" as expected behaviour, and that
#: plateau was a solver defect (floored CG ridge, Armijo below the objective's
#: float64 granularity), not a property of the cone.
_STALLED_AGREEMENT_TOLERANCE = 1e-7
#: Oracle iterations. Dykstra converges linearly; 5000 is ~25x what the former
#: production budget spent and puts the reference well past the solver.
_ORACLE_ITERATIONS = 5000
#: Minimum inflation eig-clip must show for the comparison to be meaningful.
_CLIP_INFLATION_FLOOR = 1e-8
#: A capped one-step run must be visibly short of the tolerance it was given.
_NONCONVERGENCE_FLOOR = 1e-15


# ---------------------------------------------------------------------------
# Independent oracle and fixtures
# ---------------------------------------------------------------------------


def _dykstra_oracle(
    corr: np.ndarray, iterations: int = _ORACLE_ITERATIONS
) -> np.ndarray:
    """Higham (2002) alternating projections — a *different* algorithm.

    Deliberately not imported from the production module: a parity test against
    the code under test is not evidence. This is the reference the Newton
    solver's nearest-matrix claim is checked against.
    """
    matrix = jnp.asarray(0.5 * (corr + corr.T), dtype=jnp.float64)
    zeros = jnp.zeros_like(matrix)

    def step(
        _: jax.Array, state: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        projected, iterate, correction = state
        residual = iterate - correction
        eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (residual + residual.T))
        projected = (eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        correction = projected - residual
        iterate = projected.at[jnp.diag_indices(matrix.shape[0])].set(1.0)
        return projected, iterate, correction

    return np.asarray(
        lax.fori_loop(0, iterations, step, (matrix, matrix, zeros))[1],
        dtype=np.float64,
    )


def _indefinite_cov(seed: int = 0, n: int = 5) -> np.ndarray:
    """Build a symmetric matrix with a positive diagonal and negative eigenvalue.

    Built as a Σ_dist-style matrix: positive diagonal (variances) and strong
    off-diagonals whose magnitude forces at least one negative eigenvalue.
    """
    rng = np.random.default_rng(seed)
    var = 1.0 + rng.random(n)  # positive variances
    std = np.sqrt(var)
    # Random orthogonal basis with one deliberately-negative eigenvalue, then
    # rescale to the target diagonal to mimic a non-PSD Σ_dist plug-in.
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.abs(rng.standard_normal(n)) + 0.5
    eigs[0] = -(0.05 + 0.15 * rng.random())  # small negative eigenvalue
    base = (q * eigs) @ q.T
    d = np.sqrt(np.abs(np.diag(base)))
    corr = base / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    matrix = corr * np.outer(std, std)
    w = np.linalg.eigvalsh(matrix)
    assert w.min() < 0, f"test fixture should be indefinite (min eig {w.min()})"
    return matrix


def _indefinite_cov_n(seed: int, n: int) -> np.ndarray:
    """Indefinite covariance builder for larger ``n`` (multiple negative eigs)."""
    rng = np.random.default_rng(seed)
    var = 1.0 + rng.random(n)
    std = np.sqrt(var)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.abs(rng.standard_normal(n)) + 0.5
    eigs[0] = -(0.05 + 0.15 * rng.random())
    eigs[1] = -(0.01 + 0.05 * rng.random())
    base = (q * eigs) @ q.T
    d = np.sqrt(np.abs(np.diag(base)))
    corr = base / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    matrix = corr * np.outer(std, std)
    assert np.linalg.eigvalsh(matrix).min() < 0
    return matrix


def _correlation_of(sigma: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(sigma))
    return sigma / np.outer(std, std)


def _is_psd(a: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.linalg.eigvalsh(0.5 * (a + a.T)).min() >= -tol)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(a=jax.random.normal(jax.random.PRNGKey(0), (5, 5)))
@given(a=float_arrays((5, 5), bound=3.0))
def test_ridge_psd_makes_psd(a: Array) -> None:
    """A matrix with forced negative eigenvalues comes back PSD."""
    sigma = a @ a.T - 10.0 * jnp.eye(5)  # force negative eigenvalues
    sigma_r = ridge_psd(sigma, eps=1e-6)
    eigvals = jnp.linalg.eigvalsh(sigma_r)
    eigenvalue_tolerance = -1e-5
    assert float(jnp.min(eigvals)) >= eigenvalue_tolerance


@pytest.mark.unit
@pytest.mark.property
def test_ridge_psd_already_psd() -> None:
    """An already-PSD matrix gains the ridge and nothing else."""
    sigma = jnp.eye(4) * 2.0
    sigma_r = ridge_psd(sigma, eps=1e-6)
    # Should only add eps (no negative eigvals to shift)
    assert jnp.allclose(sigma_r, sigma + 1e-6 * jnp.eye(4), atol=1e-7)


@pytest.mark.unit
@pytest.mark.property
def test_ridge_psd_symmetrises_its_input() -> None:
    """An asymmetric input must not carry its asymmetry through the repair.

    ``jnp.linalg.eigvalsh`` reads only the lower triangle, so without the
    leading symmetrisation the ridge is computed from a different matrix than
    the one returned and the asymmetry survives — leaving a downstream
    ``cholesky``/``eigh`` to silently reinterpret it. The docstring promises a
    PSD result "regardless of the input's eigenspectrum"; this is what makes
    that unconditional.
    """
    key = jax.random.PRNGKey(1)
    a = jax.random.normal(key, (5, 5))
    symmetric = 0.5 * (a + a.T)
    indefinite = symmetric - (jnp.linalg.eigvalsh(symmetric).min() + 0.5) * jnp.eye(5)
    asymmetric = indefinite.at[0, 1].add(1e-3)

    repaired = ridge_psd(asymmetric)

    assert float(jnp.linalg.norm(asymmetric - asymmetric.T)) > 0.0
    assert jnp.array_equal(repaired, repaired.T)
    assert float(jnp.min(jnp.linalg.eigvalsh(repaired))) >= _MIN_EIGENVALUE_TOLERANCE


@pytest.mark.unit
@pytest.mark.contract
def test_ridge_psd_default_eps_matches_the_shared_constant() -> None:
    """The default ridge is the shared constant, not a second opinion.

    ``ridge_psd`` and its former NumPy port ``ridge_repair`` defaulted to
    ``1e-6`` and ``1e-8`` respectively while documented as ports of one
    another. The port was deleted in t65.4 — a proven-identical duplicate is
    not a second implementation — so this now pins the surviving default.
    """
    sigma = jnp.eye(4) * 2.0
    assert pytest.approx(_EXPECTED_RIDGE_EPS, rel=0.0, abs=0.0) == DEFAULT_RIDGE_EPS
    assert jnp.allclose(
        ridge_psd(sigma), sigma + DEFAULT_RIDGE_EPS * jnp.eye(4), atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.contract
@examples(100)
@example(eps=-1.0)
@example(eps=0.0)
@example(eps=float("nan"))
@example(eps=float("inf"))
@given(
    eps=st.one_of(
        st.floats(max_value=0.0, allow_nan=False),
        st.just(float("nan")),
        st.just(float("inf")),
    )
)
def test_ridge_psd_rejects_invalid_floor(eps: float) -> None:
    """A nonpositive or nonfinite floor cannot satisfy the repair contract."""
    with pytest.raises(ValueError, match="finite and strictly positive"):
        ridge_psd(jnp.eye(2), eps=eps)


# ---------------------------------------------------------------------------
# Nearest correlation / covariance — the headline contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.reference
def test_newton_matches_an_independent_dykstra_oracle() -> None:
    """The solved matrix is *nearest*, not merely feasible.

    Checked against alternating projections at 5000 iterations — a different
    algorithm, not a second port of this one. This is the assertion the deleted
    NumPy path could not pass: at its own 200-iteration budget it sat ``7e-4``
    from the converged answer on production-shaped inputs.

    **Convergence is a precondition again, and the oracle tolerance is
    unconditional** (restored 2026-08-08). Both had been relaxed to accommodate
    a solver defect: this test previously selected ``atol`` from the solver's own
    ``converged`` leaf, so a stalled run silently moved its own accuracy bar from
    ``1e-9`` to ``1e-7``, and its docstring recorded that demanding convergence
    "made this test pass locally and fail in CI at ``4.78e-08``". That CI failure
    was real — a floored CG ridge and an Armijo test evaluated below its
    objective's float64 granularity made the verdict turn on rounding, so seed
    701 flipped under a one-ulp perturbation. Both are fixed in
    :mod:`statadic.optimize.psd`; a test that branches on the leaf it is meant to
    check cannot see either of them come back.
    """
    with jax.enable_x64(new_val=True):
        for seed in range(6):
            correlation = _correlation_of(_indefinite_cov_n(700 + seed, 20))
            oracle = _dykstra_oracle(correlation)
            result = nearest_correlation(jnp.asarray(correlation, jnp.float64))
            assert bool(result.converged), (
                f"seed {seed} did not converge "
                f"(residual {float(result.residual):.4e} at "
                f"{int(result.iterations)} steps)"
            )
            np.testing.assert_allclose(
                np.asarray(result.matrix),
                oracle,
                atol=_ORACLE_AGREEMENT_TOLERANCE,
                rtol=0.0,
                err_msg=f"seed {seed}",
            )


@pytest.mark.unit
@pytest.mark.reference
@examples(10)
@given(off_diagonal=st.lists(st.integers(-15, 15), min_size=3, max_size=3))
def test_generated_nearest_repairs_match_a_converged_correlation_space_oracle(
    off_diagonal: list[int],
) -> None:
    """Unequal marginals retain the correlation-space nearest objective."""
    correlation = np.eye(3)
    correlation[np.triu_indices(3, k=1)] = np.asarray(off_diagonal) / 10.0
    correlation += np.triu(correlation, k=1).T
    oracle = _dykstra_oracle(correlation)
    continued = _dykstra_oracle(correlation, iterations=_ORACLE_ITERATIONS + 1)
    np.testing.assert_allclose(continued, oracle, atol=1e-12, rtol=0.0)
    assert _is_psd(oracle, tol=1e-10)
    np.testing.assert_array_equal(np.diag(oracle), np.ones(3))
    deviations = np.asarray([1.0, 2.0, 3.0])
    scale = np.outer(deviations, deviations)
    correlation_result = nearest_correlation(jnp.asarray(correlation))
    covariance_result = nearest_covariance(jnp.asarray(correlation * scale))
    expected_objective = 0.5 * np.sum((oracle - correlation) ** 2)

    for result, normalization in (
        (correlation_result, np.ones((3, 3))),
        (covariance_result, scale),
    ):
        assert bool(result.converged)
        normalized = np.asarray(result.matrix) / normalization
        np.testing.assert_allclose(
            normalized, oracle, atol=_ORACLE_AGREEMENT_TOLERANCE, rtol=0.0
        )
        objective = 0.5 * np.sum((normalized - correlation) ** 2)
        assert float(objective) == pytest.approx(float(expected_objective), abs=1e-9)
        assert _is_psd(np.asarray(result.matrix), tol=1e-9)
        np.testing.assert_array_equal(np.diag(normalized), np.ones(3))


@pytest.mark.component
@pytest.mark.numerical
def test_spectral_baselines_have_distinct_known_spectrum_objectives() -> None:
    """Clipping moves bad eigenvalues only; ridge shifts the entire spectrum."""
    basis = np.linalg.qr(
        np.asarray([[1.0, 2.0, -1.0], [2.0, 0.5, 1.0], [-1.0, 1.0, 2.0]])
    )[0]
    spectrum = np.asarray([-0.5, 1.0, 3.0])
    matrix = (basis * spectrum) @ basis.T
    floor = 1e-4
    clipped = eig_clip(jnp.asarray(matrix), eps=floor)
    ridged = ridge_psd(jnp.asarray(matrix), eps=floor)

    np.testing.assert_allclose(
        clipped, (basis * np.maximum(spectrum, floor)) @ basis.T, atol=1e-12
    )
    np.testing.assert_allclose(ridged, matrix + (0.5 + floor) * np.eye(3), atol=1e-12)
    repeated_ridge = ridge_psd(ridged, eps=floor)
    np.testing.assert_allclose(repeated_ridge - ridged, floor * np.eye(3), atol=1e-12)
    assert np.linalg.norm(np.asarray(clipped) - matrix) < np.linalg.norm(
        np.asarray(ridged) - matrix
    )


@pytest.mark.unit
@pytest.mark.reference
def test_spectral_repair_derivatives_match_numpy_finite_differences_off_switches() -> (
    None
):
    """Distinct separated eigenvalues avoid rank, floor and eigenspace switches."""
    matrix = np.asarray([[2.0, 0.2, 0.1], [0.2, -0.8, 0.3], [0.1, 0.3, 1.3]])
    direction = np.asarray([[0.3, 0.1, -0.2], [0.1, -0.4, 0.05], [-0.2, 0.05, 0.2]])
    floor = 1e-4
    step = 1e-5

    def numpy_clip(value: np.ndarray) -> np.ndarray:
        """Evaluate the independent spectral projection for a smooth witness."""
        eigenvalues, vectors = np.linalg.eigh(value)
        return (vectors * np.maximum(eigenvalues, floor)) @ vectors.T

    expected = (
        numpy_clip(matrix + step * direction) - numpy_clip(matrix - step * direction)
    ) / (2 * step)

    def evaluate_eig_clip(value: jax.Array) -> Array:
        return eig_clip(value, eps=floor)

    _, actual = jax.jvp(
        evaluate_eig_clip,
        (jnp.asarray(matrix),),
        (jnp.asarray(direction),),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.component
@pytest.mark.property
@examples(20)
@example(covariance=_INDEFINITE_TRIPLE)
@given(covariance=_symmetric_matrices(4, 3.0))
def test_nearest_correlation_is_psd_with_unit_diagonal(
    covariance: np.ndarray,
) -> None:
    """Correlation repair is PSD with an exact unit diagonal."""
    with jax.enable_x64(new_val=True):
        correlation = np.asarray(
            nearest_correlation(jnp.asarray(covariance, jnp.float64)).matrix
        )
    assert _is_psd(correlation)
    assert np.array_equal(np.diag(correlation), np.ones(len(covariance)))


@pytest.mark.component
@pytest.mark.property
def test_nearest_correlation_is_idempotent_on_a_valid_correlation() -> None:
    """A valid correlation matrix is already its own nearest — a no-op."""
    covariance = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.15], [0.1, 0.15, 1.0]])
    with jax.enable_x64(new_val=True):
        result = nearest_correlation(jnp.asarray(covariance, jnp.float64))
    assert np.allclose(np.asarray(result.matrix), covariance, atol=1e-12)
    assert bool(result.converged)


@pytest.mark.component
@pytest.mark.property
def test_nearest_covariance_preserves_marginal_variances_exactly() -> None:
    """Covariance repair preserves the diagonal bit-for-bit, not approximately.

    This is why the repair runs in correlation space at all: an unweighted
    Frobenius-nearest *covariance* is free to move marginal variances, which
    for a portfolio target is the one thing it must not do.
    """
    matrix = _indefinite_cov(1, 6)
    with jax.enable_x64(new_val=True):
        repaired = np.asarray(
            nearest_covariance(jnp.asarray(matrix, jnp.float64)).matrix
        )
    assert _is_psd(repaired)
    assert np.array_equal(np.diag(repaired), np.diag(matrix))


def _original_diagonal_witness() -> np.ndarray:
    """Return the seeded matrix this test used before it was generated."""
    rng = np.random.default_rng(107)
    n = int(rng.integers(2, 15))
    matrix = rng.normal(size=(n, n))
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, rng.uniform(-5.0, 5.0, n))
    return matrix


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(matrix=_original_diagonal_witness())
@given(matrix=_symmetric_matrices(5, 5.0))
def test_nearest_correlation_ignores_the_input_diagonal(matrix: np.ndarray) -> None:
    """The constrained diagonal contributes only a constant to the objective.

    ``diag(X) = 1`` is imposed, so ``‖X - C‖²_F`` differs between two inputs
    that agree off-diagonal by a term free of ``X``. The argmin must therefore
    be identical — a property the *dual* formulation makes structural rather
    than incidental, since the diagonal of ``C`` enters only through ``y``.
    """
    unit_diagonal = matrix.copy()
    np.fill_diagonal(unit_diagonal, 1.0)

    with jax.enable_x64(new_val=True):
        repaired = np.asarray(
            nearest_correlation(jnp.asarray(matrix, jnp.float64)).matrix
        )
        expected = np.asarray(
            nearest_correlation(jnp.asarray(unit_diagonal, jnp.float64)).matrix
        )

    np.testing.assert_allclose(repaired, expected, atol=1e-11, rtol=0.0)


@pytest.mark.component
@pytest.mark.property
def test_nearest_covariance_is_cholesky_safe_20x20() -> None:
    """A repaired 20-by-20 covariance is Cholesky-safe with its diagonal intact.

    The exact nearest correlation matrix is boundary-singular, so ``cholesky``
    safety comes from the relative eigenvalue floor rather than from the
    solution. Ten seeds because a single one cannot distinguish "floored" from
    "lucky".
    """
    with jax.enable_x64(new_val=True):
        for s in range(10):
            matrix = _indefinite_cov_n(300 + s, 20)
            repaired = np.asarray(
                nearest_covariance(jnp.asarray(matrix, jnp.float64)).matrix
            )
            min_eig = float(np.linalg.eigvalsh(0.5 * (repaired + repaired.T)).min())
            assert min_eig >= -_CHOLESKY_EIGENVALUE_TOLERANCE, (
                f"min-eig {min_eig:.2e} not tight PSD"
            )
            np.linalg.cholesky(repaired)  # must not raise
            assert np.array_equal(np.diag(repaired), np.diag(matrix))


@pytest.mark.component
@pytest.mark.property
@examples(20)
@example(covariance=_INDEFINITE_TRIPLE)
@given(covariance=_symmetric_matrices(4, 3.0))
def test_nearest_correlation_is_cholesky_safe(covariance: np.ndarray) -> None:
    """Correlation repair is Cholesky-safe with an exact unit diagonal."""
    with jax.enable_x64(new_val=True):
        correlation = np.asarray(
            nearest_correlation(jnp.asarray(covariance, jnp.float64)).matrix
        )
    assert (
        float(np.linalg.eigvalsh(correlation).min()) >= -_CHOLESKY_EIGENVALUE_TOLERANCE
    )
    np.linalg.cholesky(correlation)
    assert np.array_equal(np.diag(correlation), np.ones(len(covariance)))


@pytest.mark.component
@pytest.mark.property
def test_nearest_repair_beats_clip_on_variance_inflation() -> None:
    """The nearest repair preserves total variance; eig-clip inflates it."""
    matrix = _indefinite_cov(2, 6)
    with jax.enable_x64(new_val=True):
        repaired = nearest_covariance(jnp.asarray(matrix, jnp.float64)).matrix
        clipped = eig_clip(jnp.asarray(matrix, jnp.float64))
        nearest_diagnostics = repair_diagnostics(jnp.asarray(matrix), repaired)
        clip_diagnostics = repair_diagnostics(jnp.asarray(matrix), clipped)
        trace_error = abs(
            float(
                nearest_diagnostics.total_var_after
                - nearest_diagnostics.total_var_before
            )
        )
        inflation = float(
            clip_diagnostics.total_var_after - clip_diagnostics.total_var_before
        )
    assert trace_error < _TRACE_PRESERVATION_TOLERANCE
    assert inflation > _CLIP_INFLATION_FLOOR


# ---------------------------------------------------------------------------
# Convergence evidence and iteration controls
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_capped_iteration_reports_nonconvergence_without_raising() -> None:
    """A capped budget reports evidence rather than aborting a traced caller.

    The deleted NumPy path raised ``RuntimeError`` under
    ``require_convergence=True``; nothing in production ever passed it, and an
    exception cannot cross a ``jit`` boundary. The claim is now a leaf.
    """
    covariance = np.array([[1.0, 0.9, -0.3], [0.9, 1.0, 0.9], [-0.3, 0.9, 1.0]])
    with jax.enable_x64(new_val=True):
        result = nearest_correlation(
            jnp.asarray(covariance, jnp.float64), max_newton=1, tol=1e-15
        )
    assert not bool(result.converged)
    assert float(result.residual) > _NONCONVERGENCE_FLOOR
    # …and the matrix is still feasible: a capped repair is not a broken one.
    assert np.array_equal(np.diag(np.asarray(result.matrix)), np.ones(3))
    assert _is_psd(np.asarray(result.matrix))


@pytest.mark.component
@pytest.mark.property
def test_a_too_tight_tolerance_costs_iterations_not_accuracy() -> None:
    """Retaining the best iterate is what makes the tolerance a request.

    Past the achievable residual the semismooth iteration wanders in the
    degenerate directions of the active set and the residual *rises*. Without
    the best-iterate guard, an unreachable tolerance therefore returns a worse
    matrix than a reachable one — measured 1.1e-14 against 1.1e-09 on a
    production-shaped seed while developing t65.4.

    The claim is a *comparison* between the two runs, so it is asserted as one.
    ``reachable.converged`` is deliberately not a precondition: seed 701's
    fixture is bit-sensitive to the host LAPACK (see the oracle test), and
    whether the default tolerance happens to be reachable on a given BLAS says
    nothing about whether the best-iterate guard works.
    """
    with jax.enable_x64(new_val=True):
        correlation = jnp.asarray(
            _correlation_of(_indefinite_cov_n(701, 20)), jnp.float64
        )
        oracle = _dykstra_oracle(np.asarray(correlation))
        reachable = nearest_correlation(correlation)
        unreachable = nearest_correlation(correlation, tol=1e-16, max_newton=60)
    assert not bool(unreachable.converged)
    assert int(unreachable.iterations) > int(reachable.iterations)
    # The guard's whole content: 35 extra steps past the achievable residual
    # never return a worse iterate than stopping early would have.
    assert float(unreachable.residual) <= float(reachable.residual)
    np.testing.assert_allclose(
        np.asarray(unreachable.matrix), oracle, atol=_STALLED_AGREEMENT_TOLERANCE
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_newton", 0),
        ("max_cg", 0),
        ("tol", 0.0),
        ("tol", float("nan")),
        ("eigenvalue_floor", -1.0),
    ],
)
def test_nearest_correlation_validates_its_controls(
    keyword: str, value: object
) -> None:
    """Invalid solver controls fail at trace time, before any eigendecomposition."""
    with pytest.raises(ValueError, match=r"max_newton|max_cg|tol|eigenvalue_floor"):
        nearest_correlation(jnp.eye(2), **{keyword: ill_typed(value)})


# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_the_repair_runs_at_the_callers_precision() -> None:
    """No silent promotion: a float32 caller gets float32 back.

    The deleted NumPy backend cast everything to float64 unconditionally. A
    traced repair must respect the caller's configured precision instead —
    promoting inside a library hides a policy decision the caller owns.
    """
    matrix = _indefinite_cov(9, 6).astype(np.float32)
    single = nearest_covariance(jnp.asarray(matrix))
    assert single.matrix.dtype == jnp.float32
    with jax.enable_x64(new_val=True):
        double = nearest_covariance(jnp.asarray(matrix, jnp.float64))
    assert double.matrix.dtype == jnp.float64


@pytest.mark.component
@pytest.mark.property
def test_float32_stays_feasible_and_does_not_overclaim() -> None:
    """float32 keeps the structural invariants; only the nearest claim weakens.

    The nearest correlation matrix is boundary-singular, so its smallest
    eigenvalues sit at float32 epsilon and no iteration count resolves them.
    What must not happen is a float32 run reporting ``converged=True`` while
    sitting far from the solution, or returning something infeasible.

    **The overclaim assertion is unconditional** (restored 2026-08-08). It used
    to sit inside ``if bool(result.converged):``, which is the one branch a test
    named for not overclaiming must not take: the leaf under test selected
    whether the test tested anything. It did not fire, because the stopping
    tolerance is ``n²·eps`` — 3.2e-04 in float32 — so a run whose unit diagonal
    was wrong in the fourth decimal reported ``converged=True`` by moving its own
    bar. ``psd._NEAREST_CLAIM_CEILING`` now gates the claim on an absolute
    residual as well, and this asserts that directly.
    """
    matrix = _indefinite_cov_n(311, 20).astype(np.float32)
    result = nearest_covariance(jnp.asarray(matrix))
    repaired = np.asarray(result.matrix, dtype=np.float64)
    assert np.allclose(np.diag(repaired), np.diag(matrix.astype(np.float64)), rtol=1e-6)
    assert _is_psd(repaired, tol=1e-6)
    # The nearest claim is never made at a residual float32 cannot reach.
    assert not bool(result.converged) or (
        float(result.residual) <= _NEAREST_CLAIM_CEILING
    )
    with jax.enable_x64(new_val=True):
        oracle = _dykstra_oracle(_correlation_of(matrix.astype(np.float64)))
    std = np.sqrt(np.diag(matrix.astype(np.float64)))
    np.testing.assert_allclose(
        repaired, oracle * np.outer(std, std), atol=1e-2, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_every_repair_transforms_with_a_static_result_structure() -> None:
    """Positive, indefinite, and degenerate lanes compose under JAX transforms.

    The degenerate all-ones lane matters: it is rank 1, so the dual Hessian is
    maximally singular there. A repair that only works away from the cone
    boundary is not usable inside a stage kernel.
    """
    positive = jnp.asarray([[1.0, 0.2], [0.2, 1.0]])
    indefinite = jnp.asarray([[1.0, 1.2], [1.2, 1.0]])
    degenerate = jnp.ones((2, 2))
    panels = jnp.stack((positive, indefinite, degenerate))

    results = jax.vmap(nearest_correlation)(panels)
    jax.make_jaxpr(nearest_correlation)(positive)
    jax.make_jaxpr(nearest_covariance)(positive)
    jax.make_jaxpr(eig_clip)(positive)
    jax.make_jaxpr(gerber_shrink_psd)(positive)
    jax.make_jaxpr(repair_diagnostics)(positive, positive)

    assert results.matrix.shape == (3, 2, 2)
    assert results.converged.shape == (3,)
    assert results.residual.shape == (3,)
    np.testing.assert_array_equal(
        np.diagonal(np.asarray(results.matrix), axis1=1, axis2=2),
        np.ones((3, 2)),
    )
    assert np.all(np.isfinite(np.asarray(results.matrix)))


# ---------------------------------------------------------------------------
# Gerber-inspired shrink-to-PSD
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
def test_gerber_makes_psd_preserves_diagonal() -> None:
    """Diagonal shrink reaches PSD without moving marginal variances."""
    matrix = _indefinite_cov(3, 5)
    with jax.enable_x64(new_val=True):
        result = gerber_shrink_psd(jnp.asarray(matrix, jnp.float64))
    repaired = np.asarray(result.matrix)
    assert bool(result.feasible)
    assert _is_psd(repaired)
    assert 0.0 <= float(result.alpha) <= 1.0
    assert np.allclose(np.diag(repaired), np.diag(matrix), atol=1e-10)


@pytest.mark.component
@pytest.mark.property
def test_gerber_no_shrink_when_already_psd() -> None:
    """Gerber shrinkage leaves an already-PSD matrix unchanged at ``alpha=0``."""
    matrix = np.diag([1.0, 2.0, 3.0]) + 0.05 * np.ones((3, 3))
    with jax.enable_x64(new_val=True):
        result = gerber_shrink_psd(jnp.asarray(matrix, jnp.float64))
    assert float(result.alpha) == pytest.approx(0.0, abs=0.0)
    assert bool(result.feasible)
    assert np.allclose(np.asarray(result.matrix), matrix)


@pytest.mark.component
@pytest.mark.contract
def test_gerber_reports_infeasibility_for_a_negative_variance() -> None:
    """Full diagonal shrink cannot repair a negative marginal variance.

    ``Σ(1) = diag(Σ)``, so if the diagonal itself misses the floor no intensity
    on the grid can succeed. That is data, not an exception — the former host
    implementation raised, which a traced caller cannot catch.
    """
    with jax.enable_x64(new_val=True):
        result = gerber_shrink_psd(jnp.asarray(np.diag([-1.0, 2.0]), jnp.float64))
    assert not bool(result.feasible)


@pytest.mark.component
@pytest.mark.contract
def test_gerber_honours_its_grid_cap() -> None:
    """Exhausting the shrink grid cannot jump silently to full diagonal shrink."""
    matrix = _indefinite_cov(8, 4)
    with jax.enable_x64(new_val=True):
        result = gerber_shrink_psd(jnp.asarray(matrix, jnp.float64), max_iter=0)
    assert not bool(result.feasible)
    assert float(result.alpha) == pytest.approx(0.0, abs=0.0)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("step", 0.0),
        ("step", 1.1),
        ("max_iter", -1),
        ("eps", float("nan")),
    ],
)
def test_gerber_validates_tuning_controls(keyword: str, value: object) -> None:
    """Invalid shrink-grid controls fail before the spectral sweep."""
    with pytest.raises(ValueError, match=r"step|max_iter|eps"):
        gerber_shrink_psd(jnp.eye(2), **{keyword: ill_typed(value)})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_diagnostics_reports_psd_flag_and_carries_convergence() -> None:
    """Diagnostics classify the repair and carry the solver's own evidence."""
    matrix = _indefinite_cov(5, 5)
    with jax.enable_x64(new_val=True):
        traced = jnp.asarray(matrix, jnp.float64)
        raw_diagnostics = repair_diagnostics(traced, traced)
        result = nearest_covariance(traced)
        repaired_diagnostics = repair_diagnostics(
            traced, result.matrix, converged=result.converged
        )
    assert not bool(raw_diagnostics.is_psd_after)
    assert raw_diagnostics.converged is None
    assert bool(repaired_diagnostics.is_psd_after)
    assert bool(repaired_diagnostics.converged)
    assert float(repaired_diagnostics.min_eig_after) > float(
        repaired_diagnostics.min_eig_before
    )
    assert float(repaired_diagnostics.offdiag_distortion) >= 0.0


@pytest.mark.component
@pytest.mark.contract
def test_diagnostics_psd_tolerance_is_tight() -> None:
    """PSD diagnostics use the documented tight classification tolerance.

    ``is_psd_after`` uses the documented -1e-10 tolerance, not
    a loose -1e-8 that would rubber-stamp a matrix Cholesky rejects.
    """
    n = 4
    # a matrix with min-eig ≈ -1e-9: PSD under a loose -1e-8 but NOT under -1e-10.
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))
    eigs = np.array([1e-9 * -1.0, 1.0, 2.0, 3.0])  # smallest = -1e-9
    a = (q * eigs) @ q.T
    with jax.enable_x64(new_val=True):
        d = repair_diagnostics(jnp.asarray(a, jnp.float64), jnp.asarray(a, jnp.float64))
    assert float(d.min_eig_after) < -_PSD_CLASSIFICATION_TOLERANCE
    assert not bool(d.is_psd_after)  # rejected at the tight -1e-10 tol


@pytest.mark.component
@pytest.mark.property
@pytest.mark.slow
def test_mc_nearest_preserves_variance_clip_inflates() -> None:
    """Verify variance preservation over random indefinite matrices.

    Over random indefinite matrices the nearest repair preserves total variance
    (trace) exactly while eigenvalue clipping systematically inflates it. This
    is the actual selling point over clipping (the survey's warning that
    clipping 'inflates total variance and distorts off-diagonals').
    """
    trace_error: list[float] = []
    clip_inflation: list[float] = []
    with jax.enable_x64(new_val=True):
        for s in range(30):
            matrix = jnp.asarray(_indefinite_cov(200 + s, 6), jnp.float64)
            nearest = repair_diagnostics(matrix, nearest_covariance(matrix).matrix)
            clipped = repair_diagnostics(matrix, eig_clip(matrix))
            trace_error.append(
                abs(float(nearest.total_var_after - nearest.total_var_before))
            )
            clip_inflation.append(
                float(clipped.total_var_after - clipped.total_var_before)
            )
    assert max(trace_error) < _MONTE_CARLO_TRACE_TOLERANCE
    assert np.mean(clip_inflation) > 0.0  # clip inflates on average
    assert min(clip_inflation) >= -_PSD_CLASSIFICATION_TOLERANCE
