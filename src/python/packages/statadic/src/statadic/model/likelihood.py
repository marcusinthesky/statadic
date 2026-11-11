"""Row likelihoods for log-mean Gamma, NB2 and grouped probit models."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax.scipy.special import betaln, gammaln, log_ndtr

from statadic.core.typing import Array  # noqa: TC001

__all__ = ["LikelihoodFamily", "log_likelihood_rows", "positive_poisson_log_mean"]


type LikelihoodFamily = Literal["gamma", "nb2", "probit", "positive_poisson"]


_STIRLING_THRESHOLD = 32
_LOG_SMALL_INTENSITY = -10


def _gamma_normalizer(shape: Array) -> Array:
    """Evaluate k log(k) - k - log Gamma(k) without large-k cancellation."""
    large = shape >= _STIRLING_THRESHOLD
    # Guard the inactive branch too, so differentiation never sees its
    # cancellation or overflow at extreme positive shape.
    direct_shape = jnp.where(large, _STIRLING_THRESHOLD, shape)
    direct = direct_shape * jnp.log(direct_shape) - direct_shape - gammaln(direct_shape)
    asymptotic_shape = jnp.where(large, shape, _STIRLING_THRESHOLD)
    inv = 1 / asymptotic_shape
    inv2 = inv * inv
    correction = inv * (
        -1 / 12
        + inv2 * (1 / 360 + inv2 * (-1 / 1260 + inv2 * (1 / 1680 - inv2 / 1188)))
    )
    stirling = 0.5 * (jnp.log(asymptotic_shape) - jnp.log(2 * jnp.pi)) + correction
    return jnp.where(large, stirling, direct)


def positive_poisson_log_mean(log_intensity: Array) -> Array:
    """Log of E[Y | Y > 0] for Poisson counts, including the zero-intensity limit."""
    # For tiny intensity use (1 - exp(-lambda))/lambda = 1-lambda/2+lambda²/6.
    low = jnp.exp(jnp.minimum(log_intensity, _LOG_SMALL_INTENSITY))
    small = -jnp.log1p(-low / 2 + low**2 / 6)
    # Above exp(4) the truncation correction rounds away even in float64.
    intensity = jnp.exp(jnp.clip(log_intensity, _LOG_SMALL_INTENSITY, 4.0))
    regular = log_intensity - jnp.log(-jnp.expm1(-intensity))
    return jnp.where(log_intensity < _LOG_SMALL_INTENSITY, small, regular)


def log_likelihood_rows(
    eta: Array,
    outcome: Array,
    trials: Array,
    dispersion: Array,
    family: LikelihoodFamily,
) -> Array:
    """Return complete independent-row log likelihood including constants."""
    if family == "probit":
        return (
            gammaln(trials + 1)
            - gammaln(outcome + 1)
            - gammaln(trials - outcome + 1)
            + outcome * log_ndtr(jnp.where(outcome > 0, eta, 0))
            + (trials - outcome) * log_ndtr(jnp.where(trials > outcome, -eta, 0))
        )
    if family == "positive_poisson":
        return (
            (outcome - 1) * eta
            - jnp.exp(eta)
            - gammaln(outcome + 1)
            + positive_poisson_log_mean(eta)
        )
    if family == "gamma":
        shape = 1 / dispersion
        constant = _gamma_normalizer(shape)
        log_y = jnp.log(outcome)
        z = log_y - eta
        return shape * (z - jnp.expm1(z)) - log_y + constant
    if family != "nb2":
        message = "Unsupported likelihood family."
        raise ValueError(message)
    size = 1 / dispersion
    safe_y = jnp.where(outcome > 0, outcome, 1)
    log_coefficient = -betaln(size, safe_y) - jnp.log(safe_y)
    log_ratio = jnp.where(outcome > 0, log_coefficient - outcome * jnp.log(size), 0)
    return (
        log_ratio
        + outcome * eta
        - (size + outcome) * jnp.logaddexp(0, jnp.log(dispersion) + eta)
    )
