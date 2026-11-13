"""Extreme grouped-probit likelihood tails and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.model.likelihood import log_likelihood_rows


@pytest.mark.unit
@pytest.mark.numerical
def test_probit_zero_count_tail_is_finite() -> None:
    """Absent outcomes contribute zero even when the opposite tail overflows."""
    eta = jnp.asarray([1e30, -1e30], dtype=jnp.float32)
    y = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    m = jnp.ones_like(y)

    def objective(values: jax.Array) -> jax.Array:
        return jnp.sum(log_likelihood_rows(values, y, m, jnp.asarray(1.0), "probit"))

    np.testing.assert_allclose(objective(eta), 0, atol=1e-5)
    assert jnp.all(jnp.isfinite(jax.grad(objective)(eta)))


@pytest.mark.unit
@pytest.mark.numerical
def test_nb2_tiny_dispersion_values_and_coefficient_score() -> None:
    """The beta identity preserves the finite NB2 likelihood near the boundary."""
    with jax.enable_x64(new_val=True):
        eta = jnp.zeros(3, dtype=jnp.float64)
        y = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64)
        alpha = jnp.asarray(1e-12, dtype=jnp.float64)

        def total(e: jax.Array) -> jax.Array:
            return jnp.sum(log_likelihood_rows(e, y, jnp.ones_like(y), alpha, "nb2"))

        rows = log_likelihood_rows(eta, y, jnp.ones_like(y), alpha, "nb2")
        np.testing.assert_allclose(rows, [-1.0, -1.0, -1.0 - np.log(2)], atol=1e-10)
        expected_score = y - (1 / alpha + y) * alpha / (1 + alpha)
        np.testing.assert_allclose(jax.grad(total)(eta), expected_score, atol=1e-10)


@pytest.mark.unit
@pytest.mark.reference
def test_gamma_tiny_dispersion_and_branch_derivatives() -> None:  # noqa: PLR0914
    """Stable density and joint scores agree with independent shape references."""
    import math  # noqa: PLC0415

    from scipy.special import digamma, gammaln, polygamma  # noqa: PLC0415

    with jax.enable_x64(new_val=True):
        y = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
        eta = jnp.zeros(2, dtype=jnp.float64)
        trials = jnp.ones(2, dtype=jnp.float64)
        phi = jnp.asarray(1e-15, dtype=jnp.float64)
        rows = log_likelihood_rows(eta, y, trials, phi, "gamma")
        shape = 1e15
        reference = 0.5 * (math.log(shape) - math.log(2 * math.pi)) - 1 / (12 * shape)
        np.testing.assert_allclose(rows, [reference, reference], rtol=1e-14)
        for shape in (31.999, 32.001):
            dispersion = 1 / shape
            y_scalar = jnp.asarray([1.7], dtype=jnp.float64)

            def logpdf(
                parameters: jax.Array, observed: jax.Array = y_scalar
            ) -> jax.Array:
                return jnp.sum(
                    log_likelihood_rows(
                        parameters[0:1],
                        observed,
                        jnp.ones(1),
                        parameters[1],
                        "gamma",
                    )
                )

            params = jnp.asarray([0.2, dispersion], dtype=jnp.float64)
            value = logpdf(params)
            gradient = jax.grad(logpdf)(params)
            z = math.log(1.7) - 0.2
            direct = (
                (shape - 1) * math.log(1.7)
                - 1.7 * shape * math.exp(-0.2)
                - gammaln(shape)
                + shape * (math.log(shape) - 0.2)
            )
            expected_gradient = [
                shape * math.expm1(z),
                -(shape**2) * (z - math.expm1(z) + math.log(shape) - digamma(shape)),
            ]
            np.testing.assert_allclose(value, direct, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                gradient, expected_gradient, atol=1e-9, rtol=1e-10
            )
            q = z - math.expm1(z) + math.log(shape) - digamma(shape)
            expected_hessian = [
                [-shape * math.exp(z), -(shape**2) * math.expm1(z)],
                [
                    -(shape**2) * math.expm1(z),
                    2 * shape**3 * q + shape**4 * (1 / shape - polygamma(1, shape)),
                ],
            ]
            np.testing.assert_allclose(
                jax.hessian(logpdf)(params), expected_hessian, atol=1e-7, rtol=1e-9
            )
