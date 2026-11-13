"""Independent multivariate least-squares and derivative witnesses for geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.association.geometric import centered_response_gram, geometric_projection

pytestmark = [pytest.mark.unit, pytest.mark.reference]


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5])
    x = np.asarray([0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, -2.0])
    response = np.column_stack((z + x / 2 + np.sin(z), x - z / 3 + np.cos(x)))
    return np.column_stack((np.ones_like(z), z, x)), response, np.column_stack((x, z))


def _sums(response: np.ndarray, design: np.ndarray) -> float:
    residual = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
    return float(np.sum(residual**2))


def test_centered_gram_and_partial_f_equal_multivariate_least_squares() -> None:
    """Squared Euclidean response distances recover the untruncated MANOVA trace F."""
    design, response, _ = _problem()
    reduced = design[:, :2]
    distances2 = np.sum((response[:, None] - response[None, :]) ** 2, axis=-1)
    centered = response - response.mean(axis=0)
    gram = centered_response_gram(jnp.asarray(distances2))
    np.testing.assert_allclose(gram, centered @ centered.T, atol=2e-6)
    full_basis = jnp.asarray(np.linalg.qr(design, mode="reduced")[0])
    reduced_basis = jnp.asarray(np.linalg.qr(reduced, mode="reduced")[0])
    actual = geometric_projection(gram, full_basis, reduced_basis)
    within = _sums(response, design)
    null_within = _sums(response, reduced)
    between = null_within - within
    total = float(np.sum(centered**2))
    np.testing.assert_allclose(
        [
            actual.within,
            actual.between,
            actual.total,
            actual.r_squared,
            actual.statistic,
        ],
        [within, between, total, between / null_within, between / (within / 5)],
        rtol=3e-6,
    )
    assert bool(actual.valid)
    assert int(actual.df_test) == 1
    assert int(actual.df_residual) == len(response) - design.shape[1]


def test_projection_scale_and_transformation_controls() -> None:
    """Changing arrays with fixed shapes works through JIT, vmap and scan."""
    design, response, _ = _problem()
    centered = response - response.mean(axis=0)
    gram = jnp.asarray(centered @ centered.T)
    full = jnp.asarray(np.linalg.qr(design, mode="reduced")[0])
    reduced = jnp.asarray(np.linalg.qr(design[:, :2], mode="reduced")[0])
    scales = jnp.asarray([1e-4, 1.0, 1e4])

    def evaluate(scale: jax.Array) -> jax.Array:
        result = geometric_projection(scale * gram, full, reduced)
        return jnp.stack((result.statistic, result.r_squared, result.valid))

    expected = np.repeat(
        np.asarray(evaluate(jnp.asarray(1.0)))[None, :], len(scales), axis=0
    )
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(scales), expected, rtol=1e-5)

    def step(carry: None, scale: jax.Array) -> tuple[None, jax.Array]:
        return carry, evaluate(scale)

    def scan(values: jax.Array) -> jax.Array:
        return jax.lax.scan(step, None, values)[1]

    scanned = jax.jit(scan)(scales)
    np.testing.assert_allclose(scanned, expected, rtol=1e-5)


def test_partial_f_jvp_gradient_and_hessian_equal_quadratic_ratio_derivatives() -> None:  # noqa: PLR0914 -- independent polynomial coefficients/derivatives
    """A varying response has independently computed polynomial sums of squares."""
    design, response, direction = _problem()
    full = np.linalg.qr(design, mode="reduced")[0]
    reduced = np.linalg.qr(design[:, :2], mode="reduced")[0]
    added = full @ full.T - reduced @ reduced.T
    residual = np.eye(len(response)) - full @ full.T

    def coefficients(projector: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.sum(response * (projector @ response)),
                2 * np.sum(direction * (projector @ response)),
                np.sum(direction * (projector @ direction)),
            ]
        )

    a0, a1, a2 = coefficients(added)
    b0, b1, b2 = coefficients(residual)
    point = 0.3
    numerator = a0 + a1 * point + a2 * point**2
    denominator = b0 + b1 * point + b2 * point**2
    numerator_prime, denominator_prime = a1 + 2 * a2 * point, b1 + 2 * b2 * point
    multiplier = len(response) - design.shape[1]
    expected_first = multiplier * (
        numerator_prime / denominator - numerator * denominator_prime / denominator**2
    )
    expected_second = multiplier * (
        2 * a2 / denominator
        - 2 * numerator_prime * denominator_prime / denominator**2
        - numerator * 2 * b2 / denominator**2
        + 2 * numerator * denominator_prime**2 / denominator**3
    )

    def statistic(value: jax.Array) -> jax.Array:
        moved = jnp.asarray(response) + value * jnp.asarray(direction)
        moved -= jnp.mean(moved, axis=0)
        return geometric_projection(
            moved @ moved.T, jnp.asarray(full), jnp.asarray(reduced)
        ).statistic

    value = jnp.asarray(point)
    _, tangent = jax.jvp(jax.jit(statistic), (value,), (jnp.ones_like(value),))
    assert float(tangent) == pytest.approx(expected_first, rel=2e-5)
    assert float(jax.grad(statistic)(value)) == pytest.approx(expected_first, rel=2e-5)
    assert float(jax.hessian(statistic)(value)) == pytest.approx(
        expected_second, rel=4e-5
    )


def test_projection_keeps_zero_residual_and_nonfinite_lanes_unavailable() -> None:
    """A finite-looking cancellation result cannot certify a zero residual F."""
    design, _, _ = _problem()
    full = jnp.asarray(np.linalg.qr(design, mode="reduced")[0])
    reduced = jnp.asarray(np.linalg.qr(design[:, :2], mode="reduced")[0])
    response = jnp.asarray(design[:, 1:])
    response -= jnp.mean(response, axis=0)
    gram = response @ response.T
    assert not bool(geometric_projection(gram, full, reduced).valid)
    assert not bool(
        geometric_projection(gram.at[0, 0].set(jnp.inf), full, reduced).valid
    )
