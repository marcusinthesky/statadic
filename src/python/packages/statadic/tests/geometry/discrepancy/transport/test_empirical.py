"""Unregularized empirical Wasserstein dispatch and independent LP references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog
from statadic.geometry.discrepancy.transport.empirical import (
    wasserstein_distance_jax,
    wasserstein_distance_result,
)
from statadic.geometry.discrepancy.transport.exact import (
    exact_balanced_wasserstein_jax,
)
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.measure.clouds import PackedClouds


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_unequal_weighted_wasserstein_matches_independent_lp(order: float) -> None:
    """Unequal empirical distributions attain SciPy's rectangular LP optimum."""
    x = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    y = np.asarray([[0.0, 1.0], [3.0, 2.0]])
    a, b = np.asarray([0.2, 0.3, 0.5]), np.asarray([0.4, 0.6])
    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) ** order
    constraints = np.vstack(
        [
            np.kron(np.eye(len(a)), np.ones(len(b))),
            np.kron(np.ones(len(a)), np.eye(len(b))),
        ]
    )
    oracle = linprog(
        cost.ravel(),
        A_eq=constraints,
        b_eq=np.concatenate([a, b]),
        bounds=(0, None),
        method="highs",
    )
    assert oracle.success
    left = PackedClouds.from_array(jnp.asarray(x), jnp.asarray(a))
    right = PackedClouds.from_array(jnp.asarray(y), jnp.asarray(b))
    result = jax.jit(wasserstein_distance_result)(left, right, order)
    assert bool(result.transport.input_valid & result.transport.converged)
    np.testing.assert_allclose(float(result.value) ** order, oracle.fun, rtol=4e-6)
    np.testing.assert_allclose(result.transport.plan.sum(axis=1), a, atol=2e-6)
    np.testing.assert_allclose(result.transport.plan.sum(axis=0), b, atol=2e-6)
    scalar = jax.jit(wasserstein_distance_jax)(left, right, order)
    np.testing.assert_allclose(scalar, result.value, rtol=2e-6)


@pytest.mark.component
@pytest.mark.contract
def test_uniform_dispatch_preserves_exact_assignment_target() -> None:
    """A uniform equal-size call keeps the established Hungarian distance."""
    left = PackedClouds.from_array(jnp.asarray([[0.0], [3.0], [8.0]]))
    right = PackedClouds.from_array(jnp.asarray([[1.0], [5.0], [7.0]]))
    reference = exact_balanced_wasserstein_jax(left, right, 2.0, ground=EUCLIDEAN)
    scalar = jax.jit(wasserstein_distance_jax)(left, right, 2.0)
    selected = wasserstein_distance_result(left, right, 2.0)
    np.testing.assert_allclose(scalar, reference, rtol=2e-6)
    np.testing.assert_allclose(selected.value, reference, rtol=2e-6)


@pytest.mark.component
@pytest.mark.reference
def test_uniform_unequal_supports_use_same_unregularized_target() -> None:
    """Uniform probabilities alone do not authorize an equal-size assignment."""
    left = PackedClouds.from_array(jnp.asarray([[0.0], [4.0]]))
    right = PackedClouds.from_array(jnp.asarray([[2.0]]))
    value = jax.jit(wasserstein_distance_jax)(left, right)
    np.testing.assert_allclose(value, 2.0, rtol=2e-6)


@pytest.mark.component
@pytest.mark.contract
def test_weighted_padding_and_zero_mass_coordinates_are_ignored() -> None:
    """Nonfinite unused coordinates have no influence on an empirical measure."""
    left = PackedClouds(
        jnp.asarray([[0.0], [4.0], [jnp.nan], [jnp.inf]]),
        jnp.asarray(3),
        jnp.asarray([0.75, 0.25, 0.0, 0.0]),
    )
    right = PackedClouds(jnp.asarray([[2.0], [jnp.nan]]), jnp.asarray(1))
    result = jax.jit(wasserstein_distance_result)(left, right)
    assert bool(result.transport.converged)
    np.testing.assert_allclose(result.value, 2.0, rtol=2e-6)
    np.testing.assert_array_equal(result.transport.plan[2:], 0)
    np.testing.assert_array_equal(result.transport.plan[:, 1], 0)


@pytest.mark.component
@pytest.mark.numerical
def test_selected_cost_scale_preserves_small_weighted_wasserstein() -> None:
    """Large unmatched costs cannot erase positive selected float32 costs."""
    masses = jnp.asarray([0.75, 0.25], dtype=jnp.float32)
    left = PackedClouds.from_array(
        jnp.asarray([[0.0], [1e10]], dtype=jnp.float32), masses
    )
    right = PackedClouds.from_array(
        jnp.asarray([[1e-10], [1e10]], dtype=jnp.float32), masses
    )
    result = jax.jit(wasserstein_distance_result)(left, right, 2.0)
    assert bool(result.transport.converged)
    assert bool(result.power_underflow)
    np.testing.assert_allclose(result.value, np.sqrt(0.75) * 1e-10, rtol=3e-6, atol=0)
    assert np.isnan(jax.jit(wasserstein_distance_jax)(left, right))


@pytest.mark.component
@pytest.mark.contract
def test_uniform_underflow_is_unavailable_in_scalar_workflow() -> None:
    """An assignment candidate with lost cost distinctions must not reach PCoA."""
    left = PackedClouds.from_array(jnp.asarray([[0.0], [1e10]], dtype=jnp.float32))
    right = PackedClouds.from_array(jnp.asarray([[1e-10], [1e10]], dtype=jnp.float32))
    assert np.isnan(jax.jit(wasserstein_distance_jax)(left, right))
    permuted = PackedClouds.from_array(right.data[::-1])
    assert np.isnan(jax.jit(wasserstein_distance_jax)(left, permuted))


@pytest.mark.component
@pytest.mark.contract
def test_mixed_coordinate_mass_precision_promotes_both_solver_branches() -> None:
    """Probability precision and support precision compose in a traced dispatch."""
    coordinates = jnp.asarray([[0.0], [4.0]], dtype=jnp.float32)
    masses = jnp.asarray([0.5, 0.5], dtype=jnp.float64)
    left = PackedClouds.from_array(coordinates, masses)
    right = PackedClouds.from_array(jnp.asarray([[2.0], [3.0]], dtype=jnp.float32))
    value = jax.jit(wasserstein_distance_jax)(left, right)
    assert value.dtype == jnp.float64
    np.testing.assert_allclose(value, np.sqrt(2.5), rtol=3e-6)


@pytest.mark.component
@pytest.mark.contract
def test_direct_cloud_broadcast_agrees_with_jit_and_vmap() -> None:
    """Mixed uniform/weighted lanes preserve pair dispatch under cloud broadcasting."""
    coordinates = jnp.asarray([[[0.0], [4.0]], [[0.0], [4.0]]])
    left = PackedClouds.from_array(coordinates, jnp.asarray([[0.5, 0.5], [0.75, 0.25]]))
    right = PackedClouds.from_array(jnp.asarray([[2.0], [3.0]]))
    direct = wasserstein_distance_jax(left, right)
    compiled = jax.jit(wasserstein_distance_jax)(left, right)
    mapped = jax.jit(jax.vmap(wasserstein_distance_jax, (0, None)))(left, right)
    expected = [np.sqrt(2.5), np.sqrt(4.5)]
    for result in (direct, compiled, mapped):
        np.testing.assert_allclose(result, expected, rtol=3e-6)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("order", [0.5, np.inf, np.nan])
def test_invalid_numeric_order_is_unavailable_under_jit(order: float) -> None:
    """Finite metric orders remain required for the rooted transport target."""
    left = PackedClouds.from_array(
        jnp.asarray([[0.0], [4.0]]), jnp.asarray([0.75, 0.25])
    )
    right = PackedClouds.from_array(jnp.asarray([[2.0]]))
    result = jax.jit(wasserstein_distance_result)(left, right, order)
    assert not bool(result.transport.input_valid)
    assert np.isnan(result.value)
