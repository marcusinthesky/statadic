"""Native point distances and RBF kernels preserve locally resolved differences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.ground.metrics import squared_euclidean_distance
from statadic.geometry.ground.similarities import RBFKernel, gram, rbf_kernel

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import DTypeLike


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize(("dtype", "offset"), [(jnp.float32, 1e4), (jnp.float64, 1e10)])
def test_broad_spread_point_distance_retains_local_difference(
    dtype: DTypeLike, offset: float
) -> None:
    """A single common-origin shift cannot repair this broad-spread witness."""
    x = jnp.asarray([[0.0, 0.0], [offset, 0.0]], dtype=dtype)
    y = jnp.asarray([[offset + 1.0, 0.0]], dtype=dtype)
    expected = np.sum((np.asarray(x)[:, None] - np.asarray(y)[None, :]) ** 2, axis=-1)
    actual = squared_euclidean_distance(x, y)
    np.testing.assert_allclose(
        actual, expected, rtol=2e-7 if dtype == jnp.float32 else 1e-15
    )
    np.testing.assert_array_equal(actual[1, 0], 1.0)


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("dynamic", [False, True])
def test_both_rbf_factories_preserve_large_translation(*, dynamic: bool) -> None:
    """Static declarations and dynamic pytrees use the same stable primitive."""
    kernel = RBFKernel(jnp.asarray(1.0)) if dynamic else rbf_kernel(1.0)
    x = jnp.asarray([[1e10]])
    y = jnp.asarray([[1e10 + 1.0]])
    actual = gram(x, y, kernel)
    np.testing.assert_allclose(actual, [[np.exp(-0.5)]], rtol=1e-14, atol=1e-14)


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("coincident", [False, True])
def test_squared_distance_and_rbf_gradients_match_analytic_scores(
    *, coincident: bool
) -> None:
    """Direct differences retain finite exact zero gradients at coincidence."""
    y = jnp.asarray([[1e10, 2.0]])
    x = y if coincident else y + jnp.asarray([[1.0, -2.0]])

    def evaluate_total(points: jax.Array) -> jax.Array:
        return squared_euclidean_distance(points, y).sum()

    value, gradient = jax.jit(jax.value_and_grad(evaluate_total))(x)
    gap = np.asarray(x) - np.asarray(y)
    np.testing.assert_allclose(value, np.square(gap).sum(), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gradient, 2 * gap, rtol=1e-14, atol=1e-14)
    kernel = RBFKernel(jnp.asarray(1.7))

    def evaluate_rbf_gram_sum(points: jax.Array) -> Array:
        return gram(points, y, kernel).sum()

    derivative = jax.jit(jax.grad(evaluate_rbf_gram_sum))(x)
    expected = -gap / 1.7**2 * np.exp(-np.square(gap).sum() / (2 * 1.7**2))
    np.testing.assert_allclose(derivative, expected, rtol=1e-13, atol=1e-13)
