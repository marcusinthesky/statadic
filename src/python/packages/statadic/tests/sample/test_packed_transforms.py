"""Empirical distributions compose with native JAX transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import wasserstein_distance
from statadic.discrepancy.mmd.scalar import mmd_squared
from statadic.discrepancy.transport.univariate import wasserstein_1d
from statadic.sample.packed import PackedClouds


@pytest.mark.integration
@pytest.mark.contract
def test_cloud_axes_are_native_pytree_batch_axes() -> None:
    """Vmap selects individual ragged distributions without losing masses."""
    clouds = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [4.0]]), jnp.asarray([[7.0]])],
        [jnp.asarray([0.25, 0.75]), jnp.asarray([1.0])],
    )
    np.testing.assert_allclose(jax.jit(jax.vmap(PackedClouds.mean))(clouds), [[3], [7]])
    assert len(jax.tree.leaves(clouds)) == len(
        (clouds.data, clouds.counts, clouds.masses)
    )

    # Shape trees exercise reconstruction with non-array placeholder leaves.
    def shape(array: jax.Array) -> tuple[int, ...]:
        return array.shape

    shapes = jax.tree.map(shape, clouds)
    assert shapes.data == (2, 2, 1)


@pytest.mark.integration
@pytest.mark.numerical
def test_coordinate_and_logit_gradients_have_independent_witnesses() -> None:
    """Linear MMD derivatives agree with the weighted-mean identity."""
    cloud = PackedClouds.from_clouds(
        [jnp.asarray([[1.0], [3.0]]), jnp.asarray([[5.0]])]
    )
    target = PackedClouds.from_points(jnp.asarray([[0.0]]))
    params = {"coordinates": cloud.coordinates, "logits": jnp.zeros_like(cloud.masses)}

    def loss(parameters: dict[str, jax.Array]) -> jax.Array:
        current = cloud.with_coordinates(parameters["coordinates"]).with_mass_logits(
            parameters["logits"]
        )
        return jnp.sum(mmd_squared(current, target))

    value, gradient = jax.jit(jax.value_and_grad(loss))(params)
    np.testing.assert_allclose(value, 29)
    np.testing.assert_allclose(gradient["coordinates"], [[[2], [2]], [[10], [0]]])
    np.testing.assert_allclose(gradient["logits"], [[-2, 2], [0, 0]])
    _, action = jax.linearize(jax.grad(loss), params)
    direction = jax.tree.map(jnp.ones_like, params)
    result = action(direction)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(result))


@pytest.mark.integration
@pytest.mark.property
def test_padding_is_inert_for_values_and_coordinate_gradients() -> None:
    """Inactive NaNs are eliminated before the numerical graph."""
    cloud = PackedClouds.from_clouds(
        [jnp.asarray([[1.0], [3.0]]), jnp.asarray([[5.0]])]
    )
    target = PackedClouds.from_points(jnp.asarray([[0.0]]))
    data = cloud.data.at[1, 1, 0].set(jnp.nan)

    def objective(coordinates: jax.Array) -> jax.Array:
        return jnp.sum(mmd_squared(cloud.with_coordinates(coordinates), target))

    assert float(objective(data)) == pytest.approx(29)
    derivative = jax.grad(objective)(data)
    assert np.isfinite(derivative).all()
    assert derivative[1, 1, 0] == 0


@pytest.mark.unit
@pytest.mark.reference
def test_weighted_ragged_w1_matches_scipy() -> None:
    """The probability carrier preserves unequal-support weighted transport."""
    left = PackedClouds.from_points(jnp.asarray([[0.0], [4.0]]), [0.25, 0.75])
    right = PackedClouds.from_points(
        jnp.asarray([[1.0], [2.0], [7.0]]), [0.2, 0.3, 0.5]
    )
    expected = wasserstein_distance([0, 4], [1, 2, 7], [0.25, 0.75], [0.2, 0.3, 0.5])
    assert float(
        jax.jit(wasserstein_1d, static_argnames=("p",))(left, right, p=1)
    ) == pytest.approx(expected)
