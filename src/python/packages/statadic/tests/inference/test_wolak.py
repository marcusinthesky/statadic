# jaxtyping shape strings are runtime contracts.
"""Tests for :mod:`statadic.inference.wolak` — chi-bar-squared and the Wolak test."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.inference.wolak import (
    chi_bar_squared_sf,
    chi_bar_squared_weights_mc,
    chi_bar_squared_weights_mc_kernel,
    wolak_statistic,
    wolak_test,
    wolak_test_kernel,
)
from scipy import stats

from .._budgets import examples
from .._strategies import prng_keys
from ._moments import (
    TAIL_CALIBRATION_TOLERANCE,
    WOLAK_DECISION_MARGIN,
)

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from jax import Array
    from statadic.inference.wolak import (
        WolakKernelResult,
    )

# ---------------------------------------------------------------------------
# chi_bar_squared
# ---------------------------------------------------------------------------


def covariances(size: int = 2) -> SearchStrategy[np.ndarray]:
    """Draw well-conditioned covariance matrices ``A A' + I / 2`` by construction.

    Args:
        size: Matrix dimension.

    Returns:
        A strategy for symmetric positive definite NumPy matrices.

    """
    return hnp.arrays(np.float64, (size, size), elements=st.floats(-1.0, 1.0)).map(
        lambda a: a @ a.T + 0.5 * np.eye(size)
    )


def _active_set_projection(targets: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Enumerate every orthant face independently for tiny strictly convex QPs."""
    best = np.zeros_like(targets)
    objective = np.full(len(targets), np.inf)
    tolerance = 1e-10
    for face in itertools.product((False, True), repeat=precision.shape[0]):
        free = np.flatnonzero(face)
        candidate = np.zeros_like(targets)
        if len(free):
            candidate[:, free] = np.linalg.solve(
                precision[np.ix_(free, free)], (targets @ precision)[:, free].T
            ).T
        gradient = (candidate - targets) @ precision
        feasible = (candidate >= -tolerance).all(1)
        feasible &= (gradient[:, np.logical_not(face)] >= -tolerance).all(1)
        difference = candidate - targets
        loss = np.einsum("bi,ij,bj->b", difference, precision, difference)
        improves = feasible & (loss < objective)
        best[improves], objective[improves] = candidate[improves], loss[improves]
    assert np.isfinite(objective).all()
    return best


def _empirical_min_distance_stats(
    covariance: np.ndarray, *, draw_count: int, seed: int
) -> np.ndarray:
    """Draw the empirical null of min_{t>=0}(t-Z)'covariance^-1(t-Z)."""
    cholesky: np.ndarray = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)
    standard_normals: np.ndarray = rng.standard_normal(
        (draw_count, covariance.shape[0])
    )
    targets: np.ndarray = (cholesky @ standard_normals.T).T
    covariance_inverse: np.ndarray = np.linalg.pinv(covariance)
    projection = _active_set_projection(targets, covariance_inverse)
    difference: np.ndarray = targets - projection
    return np.einsum(
        "bi,ij,bj->b",
        difference,
        covariance_inverse,
        difference,
    )


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(mean=[-0.8, 0.4], covariance=np.asarray([[1.0, 0.55], [0.55, 1.4]]))
@example(mean=[0.3, -0.6], covariance=np.asarray([[1.0, 0.55], [0.55, 1.4]]))
@example(mean=[-0.2, -0.7], covariance=np.asarray([[1.0, 0.55], [0.55, 1.4]]))
@example(mean=[0.4, 0.6], covariance=np.asarray([[1.0, 0.55], [0.55, 1.4]]))
@example(mean=[0.0, 1e-9], covariance=0.5 * np.eye(2))
@given(
    mean=st.lists(st.floats(-1.0, 1.0), min_size=2, max_size=2),
    covariance=covariances(),
)
def test_wolak_statistic_matches_independent_active_set_optimum(
    mean: list[float], covariance: np.ndarray
) -> None:
    """A correlated metric requires projection, not coordinatewise clipping.

    Args:
        mean: Generated moment means.
        covariance: Generated positive definite covariance.

    """
    precision = np.linalg.inv(covariance)
    target = np.asarray(mean)
    projection = _active_set_projection(target[None, :], precision)[0]
    difference = target - projection
    count = 37

    statistic, binding = wolak_statistic(target, covariance, count)

    np.testing.assert_allclose(
        statistic, count * difference @ precision @ difference, rtol=1e-10, atol=1e-12
    )
    # Binding is the solver's absolute 1e-10 numerical boundary, independently
    # applied to the face-enumerated projection. A positive 1e-9 is free.
    binding_tolerance = 1e-10
    assert binding == np.count_nonzero(np.abs(projection) <= binding_tolerance)


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(key=jax.random.PRNGKey(23), covariance=np.asarray([[1.0, 0.2], [0.2, 1.0]]))
@given(key=prng_keys(), covariance=covariances())
def test_keyed_wolak_kernels_are_deterministic_array_pytrees(
    key: Array, covariance: np.ndarray
) -> None:
    """The t66 kernels trace from explicit keys and expose fixed array shapes.

    Args:
        key: Generated PRNG key.
        covariance: Generated positive definite covariance.

    """
    traced = jnp.asarray(covariance)
    first = chi_bar_squared_weights_mc_kernel(key, traced, n_mc=32)
    second = chi_bar_squared_weights_mc_kernel(key, traced, n_mc=32)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.weights.shape == (3,)
    assert first.converged.shape == ()

    time = jnp.linspace(0.0, 5.0, 64)
    moments = jnp.stack((0.1 + jnp.sin(time), 0.2 + jnp.cos(time)), axis=1)
    result = wolak_test_kernel(key, moments, bandwidth=1, n_mc=32)
    assert result.stat.shape == ()
    assert result.pvalue.shape == ()
    assert result.weights.shape == (3,)
    assert result.n_binding.shape == ()
    assert result.converged.shape == ()

    def evaluate_wolak_test_kernel(explicit_key: jax.Array) -> WolakKernelResult:
        return wolak_test_kernel(explicit_key, moments, bandwidth=1, n_mc=32)

    jax.make_jaxpr(evaluate_wolak_test_kernel)(key)


@pytest.mark.component
@pytest.mark.property
def test_wolak_kernel_sweep_budget_is_a_result_invariant() -> None:
    """A lower converged sweep budget preserves weights, p-values, and decisions."""
    key = jax.random.PRNGKey(29)
    time = jnp.linspace(0.0, 5.0, 64)
    moments = jnp.stack((-0.1 + jnp.sin(time), 0.2 + jnp.cos(time)), axis=1)
    reference = wolak_test_kernel(key, moments, bandwidth=1, n_mc=64)
    reduced = wolak_test_kernel(
        key,
        moments,
        bandwidth=1,
        n_mc=64,
        max_sweeps=256,
    )
    assert bool(reference.converged)
    assert bool(reduced.converged)
    np.testing.assert_array_equal(reduced.weights, reference.weights)
    np.testing.assert_array_equal(reduced.pvalue, reference.pvalue)
    assert bool(reduced.pvalue < WOLAK_DECISION_MARGIN) == bool(
        reference.pvalue < WOLAK_DECISION_MARGIN
    )


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_chi_bar_squared_weights_mc_identity() -> None:
    """Identity covariance yields the expected chi-bar-squared Monte Carlo weights."""
    covariance = np.eye(2)
    w = chi_bar_squared_weights_mc(covariance, n_mc=2000, seed=0)
    assert w.shape == (3,)
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, [0.25, 0.5, 0.25], atol=0.05)


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.parametrize("m", [3, 4])
def test_chi_bar_squared_weights_calibration(m: int) -> None:
    """Verify chi-bar-squared calibration against an empirical null.

    The χ̄² weight for χ²_k must use exactly ``k`` binding projected
    (zero) projected coords), i.e. the mixture dof = number of binding coords,
    NOT the number of strictly-positive coords.  Verify by matching the fitted
    χ̄² tail to the empirical null distribution of the min-distance statistic
    min_{t≥0}(t-Z)'covariance⁻¹(t-Z) at m=3/m=4, rho=0.6.  Under the reversed (buggy)
    indexing the fitted tail at x=1 was ≈0.577 (m=3) vs the true ≈0.31.
    """
    rho = 0.6
    covariance = (1.0 - rho) * np.eye(m) + rho * np.ones((m, m))
    weights = chi_bar_squared_weights_mc(covariance, n_mc=40000, seed=0)
    assert weights.shape == (m + 1,)
    assert np.isclose(weights.sum(), 1.0)

    # Empirical null tail of the min-distance statistic for gbar ~ N(0, covariance).
    emp_stats = _empirical_min_distance_stats(covariance, draw_count=60000, seed=1)

    for x in (0.5, 1.0, 2.0):
        fitted = chi_bar_squared_sf(x, weights)
        empirical = float(np.mean(emp_stats > x))
        assert abs(fitted - empirical) < TAIL_CALIBRATION_TOLERANCE, (
            f"χ̄² miscalibrated at m={m}, x={x}: fitted {fitted:.4f} "
            f"vs empirical {empirical:.4f}"
        )

    # The buggy reversed weights would give a markedly larger tail at x=1.
    reversed_tail = chi_bar_squared_sf(1.0, weights[::-1])
    fitted_tail = chi_bar_squared_sf(1.0, weights)
    assert reversed_tail > fitted_tail + 0.1


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@examples(100)
@given(
    stat=st.floats(0.01, 30.0),
    raw=st.lists(st.floats(0.01, 1.0), min_size=3, max_size=3),
)
def test_chi_bar_squared_sf_is_the_weighted_chi2_mixture(
    stat: float, raw: list[float]
) -> None:
    """The tail is the weight-averaged chi-square tail; the zero-df atom has none.

    Args:
        stat: Generated positive statistic.
        raw: Unnormalised generated mixture weights.

    """
    weights = np.asarray(raw) / np.sum(raw)
    expected = sum(weights[k] * stats.chi2.sf(stat, k) for k in range(1, len(weights)))
    np.testing.assert_allclose(
        chi_bar_squared_sf(stat, weights), expected, rtol=1e-8, atol=1e-12
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_chi_bar_squared_sf_pure_chi2() -> None:
    """Pinned pure chi-square and pure-atom cases."""
    stat = 4.3
    assert np.isclose(
        chi_bar_squared_sf(stat, np.array([0.0, 0.0, 1.0])), stats.chi2.sf(stat, 2)
    )
    np.testing.assert_equal(chi_bar_squared_sf(stat, np.array([1.0, 0.0, 0.0])), 0.0)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(n_mc=0, scale=1.0, bad=float("nan"))
@given(
    n_mc=st.integers(max_value=0),
    scale=st.floats(0.1, 10.0),
    bad=st.sampled_from([float("nan"), float("inf"), float("-inf")]),
)
def test_wolak_calibration_rejects_invalid_covariance_and_draw_count(
    n_mc: int, scale: float, bad: float
) -> None:
    """A chi-bar reference needs positive draws and all stochastic directions.

    Args:
        n_mc: Any non-positive draw count.
        scale: Scale of a rank-one (singular) covariance.
        bad: A nonfinite moment entry.

    """
    with pytest.raises(ValueError, match="positive integer"):
        chi_bar_squared_weights_mc(np.eye(2), n_mc=n_mc)
    with pytest.raises(ValueError, match="full rank"):
        chi_bar_squared_weights_mc(scale * np.ones((2, 2)), n_mc=10)
    with pytest.raises(ValueError, match="finite"):
        wolak_test(np.array([[0.0], [bad]]), n_mc=10)


# ---------------------------------------------------------------------------
# wolak_test
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
@pytest.mark.slow
def test_wolak_test_h0_and_violation() -> None:
    """Wolak testing separates a feasible null from an inequality violation."""
    rng = np.random.default_rng(5)
    g_ok = 1.0 + 0.3 * rng.standard_normal((300, 2))  # moments >= 0 satisfied
    res_ok = wolak_test(g_ok, n_mc=1000, seed=0)
    assert res_ok.pvalue > WOLAK_DECISION_MARGIN

    g_bad = 1.0 + 0.3 * rng.standard_normal((300, 2))
    g_bad[:, 0] -= 2.0  # one moment mean ≈ -1 → violated inequality
    res_bad = wolak_test(g_bad, n_mc=1000, seed=0)
    assert res_bad.pvalue < WOLAK_DECISION_MARGIN
