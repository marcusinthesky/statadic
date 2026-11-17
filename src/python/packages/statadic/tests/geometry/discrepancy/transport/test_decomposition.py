"""Euclidean squared-W2 location and centered-shape identity witnesses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.discrepancy.transport.decomposition import (
    euclidean_wasserstein2_decomposition,
)
from statadic.measure.clouds import PackedClouds


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pure_translation_is_location_only(dtype: type) -> None:
    """Translation changes the centroid by the shift and leaves centered shape."""
    if dtype is jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 witness requires caller-enabled x64")
    points = jnp.asarray([[-2.0, 1.0], [0.0, -1.0], [3.0, 2.0]], dtype=dtype)
    left = PackedClouds.from_array(points)
    right = PackedClouds.from_array(points + jnp.asarray([3.0, -4.0], dtype=dtype))
    result = jax.jit(euclidean_wasserstein2_decomposition)(left, right)
    tolerance = 64 * np.finfo(np.dtype(dtype)).eps
    assert bool(result.finite)
    np.testing.assert_allclose(result.distance_squared, 25.0, rtol=tolerance)
    np.testing.assert_allclose(result.location_squared, 25.0, rtol=tolerance)
    np.testing.assert_allclose(result.shape_squared, 0.0, atol=tolerance)
    np.testing.assert_allclose(result.reconstruction_error, 0.0, atol=25 * tolerance)
    assert result.distance_squared.dtype == points.dtype


@pytest.mark.unit
@pytest.mark.reference
def test_equal_means_preserve_nonzero_shape() -> None:
    """Centroids coincide while article distributions remain one unit apart."""
    left = PackedClouds.from_array(jnp.asarray([[-1.0], [1.0]]))
    right = PackedClouds.from_array(jnp.asarray([[-2.0], [2.0]]))
    result = euclidean_wasserstein2_decomposition(left, right)
    assert bool(result.finite)
    np.testing.assert_allclose(result.distance_squared, 1.0)
    np.testing.assert_allclose(result.location_squared, 0.0)
    np.testing.assert_allclose(result.shape_squared, 1.0)
    np.testing.assert_allclose(result.reconstruction_error, 0.0, atol=1e-6)


@pytest.mark.component
@pytest.mark.reference
def test_weighted_unequal_supports_decompose_with_padding() -> None:
    """A Dirac target has analytic transport cost even with unequal weighted support."""
    left = PackedClouds(
        jnp.asarray([[0.0], [4.0], [jnp.nan]]),
        jnp.asarray(2),
        jnp.asarray([0.75, 0.25, 0.0]),
    )
    right = PackedClouds.from_array(jnp.asarray([[2.0]]))
    result = jax.jit(euclidean_wasserstein2_decomposition)(left, right)
    assert bool(result.finite)
    np.testing.assert_allclose(result.distance_squared, 4.0, rtol=2e-6)
    np.testing.assert_allclose(result.location_squared, 1.0, rtol=2e-6)
    np.testing.assert_allclose(result.shape_squared, 3.0, rtol=2e-6)
    np.testing.assert_allclose(result.reconstruction_error, 0.0, atol=2e-6)


@pytest.mark.component
@pytest.mark.contract
def test_decomposition_broadcasts_and_agrees_with_vmap() -> None:
    """One common reference and batched empirical measures have identical semantics."""
    points = jnp.asarray([[[-1.0], [1.0]], [[0.0], [2.0]]])
    left = PackedClouds.from_array(points)
    right = PackedClouds.from_array(points[0])
    observed = jax.jit(euclidean_wasserstein2_decomposition)(left, right)
    mapped = jax.jit(jax.vmap(euclidean_wasserstein2_decomposition, (0, None)))(
        left, right
    )
    for actual, expected in zip(
        jax.tree.leaves(observed), jax.tree.leaves(mapped), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(observed.distance_squared, [0.0, 1.0], atol=2e-6)
    np.testing.assert_allclose(observed.location_squared, [0.0, 1.0], atol=2e-6)
    np.testing.assert_allclose(observed.shape_squared, [0.0, 0.0], atol=2e-6)


@pytest.mark.unit
@pytest.mark.contract
def test_invalid_probability_measure_has_nonfinite_diagnostic() -> None:
    """Numeric probability failures remain visible under transformed diagnostics."""
    left = PackedClouds(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray(2),
        jnp.asarray([0.4, 0.4]),
    )
    right = PackedClouds.from_array(jnp.asarray([[0.0], [1.0]]))
    result = jax.jit(euclidean_wasserstein2_decomposition)(left, right)
    assert not bool(result.finite)
    assert np.isnan(result.distance_squared)
    assert np.isnan(result.shape_squared)
