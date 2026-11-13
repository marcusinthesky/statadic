"""Norm derivatives use independent analytic controls and explicit collision limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from statadic.geometry.ground.metrics import (
    euclidean_distance,
    normalized_euclidean_distance,
    sphere_chord_distance,
)

if TYPE_CHECKING:
    from jax import Array


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_euclidean_coincidence_has_selected_zero_first_derivative(dtype: str) -> None:
    """Self-pairs never multiply an infinite sqrt derivative by zero."""
    with jax.enable_x64(new_val=dtype == "float64"):
        origin = jnp.zeros((1, 2), dtype=dtype)
        direction = jnp.asarray([[0.3, -0.8]], dtype=dtype)

        def objective(value: Array) -> Array:
            return euclidean_distance(value, origin)[0, 0]

        def derivative(value: Array, tangent: Array) -> tuple[Array, Array]:
            return jax.jvp(objective, (value,), (tangent,))

        value, tangent = jax.jit(derivative)(origin, direction)
        np.testing.assert_array_equal(value, 0)
        np.testing.assert_array_equal(tangent, 0)
        np.testing.assert_array_equal(jax.jit(jax.grad(objective))(origin), origin)
        np.testing.assert_array_equal(jax.jit(jax.vmap(objective))(origin[None]), [0])


@pytest.mark.unit
@pytest.mark.numerical
def test_euclidean_away_from_collision_matches_gradient_and_hessian() -> None:
    """The ordinary norm has gradient u and Hessian (I-u*u')/r."""
    point, direction = jnp.asarray([3.0, 4.0]), jnp.asarray([2.0, -1.0])
    origin = jnp.zeros((1, 2))

    def objective(value: Array) -> Array:
        return euclidean_distance(value[None], origin)[0, 0]

    gradient = np.asarray([0.6, 0.8])
    hessian = (np.eye(2) - np.outer(gradient, gradient)) / 5
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(point), gradient, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(objective))(point), hessian, atol=1e-12
    )
    _, hvp = jax.jvp(jax.grad(objective), (point,), (direction,))
    np.testing.assert_allclose(hvp, hessian @ direction, atol=1e-12)

    def accumulated(value: Array) -> Array:
        def step(total: Array, multiplier: Array) -> tuple[Array, Array]:
            distance = objective(value * multiplier)
            return total + distance, distance

        return lax.scan(step, jnp.zeros(()), jnp.asarray([1.0, 2.0]))[0]

    np.testing.assert_allclose(jax.jit(accumulated)(point), 15, atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(jax.grad(accumulated))(point), 3 * gradient, atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.regression
def test_chord_collision_selection_preserves_nonfinite_domain_failures() -> None:
    """Coincident nonzero/unit rows have finite first derivatives; NaN stays NaN."""
    point = jnp.asarray([[1.0, 0.0]])

    def normalized(value: Array) -> Array:
        return normalized_euclidean_distance(value, point)[0, 0]

    def sphere(value: Array) -> Array:
        return sphere_chord_distance(value, point)[0, 0]

    np.testing.assert_array_equal(
        jax.jit(jax.grad(normalized))(point), jnp.zeros_like(point)
    )
    np.testing.assert_array_equal(
        jax.jit(jax.grad(sphere))(point), jnp.zeros_like(point)
    )
    assert np.isnan(euclidean_distance(jnp.asarray([[jnp.nan, 0.0]]), point)).all()
    assert np.isnan(normalized_euclidean_distance(jnp.zeros_like(point), point)).all()
