"""Independent parametric accelerated-failure-time equations and predictions.

The ancillary parameter is constant. Duration and entry are elapsed spell times,
not calendar dates. Right-censoring and entry are conditionally noninformative.
Numerical functions support JAX transforms; host data alignment is separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln, log_ndtr, ndtri

from statadic.core.typing import Array, ArrayLike  # noqa: TC001


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WeibullAFT:
    """Weibull AFT with log scale ``X beta`` and log shape as ancillary."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LogNormalAFT:
    """Lognormal AFT with log-time location ``X beta`` and log standard deviation."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LogLogisticAFT:
    """Log-logistic AFT with log scale ``X beta`` and log shape."""


type AFTFamily = WeibullAFT | LogNormalAFT | LogLogisticAFT


def _location(parameters: ArrayLike, design: ArrayLike) -> tuple[Array, Array]:
    theta, x = jnp.asarray(parameters), jnp.asarray(design)
    return jnp.matmul(x, theta[:-1]), theta[-1]


def log_survival(
    parameters: ArrayLike, design: ArrayLike, time: ArrayLike, family: AFTFamily
) -> Array:
    """Log P(T > time), including the continuous-law value 0 at time zero."""
    eta, ancillary = _location(parameters, design)
    t = jnp.asarray(time)
    safe_time = jnp.where(t > 0, t, 1)
    log_t = jnp.log(safe_time)
    if isinstance(family, WeibullAFT):
        value = -jnp.exp(jnp.exp(ancillary) * (log_t - eta))
    elif isinstance(family, LogNormalAFT):
        value = log_ndtr(-(log_t - eta) * jnp.exp(-ancillary))
    elif isinstance(family, LogLogisticAFT):
        value = -jax.nn.softplus(jnp.exp(ancillary) * (log_t - eta))
    else:
        msg = "Unsupported AFT family."
        raise TypeError(msg)
    return jnp.where(jnp.isfinite(t) & (t >= 0), jnp.where(t == 0, 0, value), jnp.nan)


def log_density(
    parameters: ArrayLike, design: ArrayLike, time: ArrayLike, family: AFTFamily
) -> Array:
    """Log event density in original duration units, including its Jacobian."""
    eta, ancillary = _location(parameters, design)
    t = jnp.asarray(time)
    safe_time = jnp.where(t > 0, t, 1)
    log_t = jnp.log(safe_time)
    if isinstance(family, WeibullAFT):
        z = jnp.exp(ancillary) * (log_t - eta)
        value = ancillary - log_t + z - jnp.exp(z)
    elif isinstance(family, LogNormalAFT):
        z = (log_t - eta) * jnp.exp(-ancillary)
        value = -log_t - ancillary - z * z / 2 - math.log(2 * math.pi) / 2
    elif isinstance(family, LogLogisticAFT):
        z = jnp.exp(ancillary) * (log_t - eta)
        value = ancillary - log_t - jax.nn.softplus(-z) - jax.nn.softplus(z)
    else:
        msg = "Unsupported AFT family."
        raise TypeError(msg)
    return jnp.where(jnp.isfinite(t) & (t > 0), value, jnp.nan)


def log_likelihood_rows(
    parameters: ArrayLike,
    design: ArrayLike,
    duration: ArrayLike,
    event: ArrayLike,
    entry: ArrayLike,
    family: AFTFamily,
) -> Array:
    """Per-spell event/censor log likelihood conditional on survival to entry."""
    delta = jnp.asarray(event)
    contribution = jnp.where(
        delta == 1,
        log_density(parameters, design, duration, family),
        log_survival(parameters, design, duration, family),
    )
    return contribution - log_survival(parameters, design, entry, family)


def survival_probability(
    parameters: ArrayLike,
    design: ArrayLike,
    time: ArrayLike,
    family: AFTFamily,
    *,
    entry: ArrayLike | None = None,
) -> Array:
    """Marginal survival, or survival conditional on lasting to ``entry``."""
    t = jnp.asarray(time)
    log_s = log_survival(parameters, design, t, family)
    if entry is not None:
        a = jnp.asarray(entry)
        log_s = jnp.where(
            (a >= 0) & (t >= a),
            log_s - log_survival(parameters, design, a, family),
            jnp.nan,
        )
    return jnp.exp(log_s)


def cumulative_hazard(
    parameters: ArrayLike,
    design: ArrayLike,
    time: ArrayLike,
    family: AFTFamily,
    *,
    entry: ArrayLike | None = None,
) -> Array:
    """Marginal or entry-conditional cumulative hazard."""
    t = jnp.asarray(time)
    value = -log_survival(parameters, design, t, family)
    if entry is not None:
        a = jnp.asarray(entry)
        value = jnp.where(
            (a >= 0) & (t >= a),
            value + log_survival(parameters, design, a, family),
            jnp.nan,
        )
    return value


def survival_quantile(
    parameters: ArrayLike, design: ArrayLike, probability: ArrayLike, family: AFTFamily
) -> Array:
    """Return the quantile for a CDF probability strictly between zero and one."""
    eta, ancillary = _location(parameters, design)
    q = jnp.asarray(probability)
    safe_q = jnp.where((q > 0) & (q < 1), q, 0.5)
    if isinstance(family, WeibullAFT):
        value = jnp.exp(eta + jnp.log(-jnp.log1p(-safe_q)) * jnp.exp(-ancillary))
    elif isinstance(family, LogNormalAFT):
        value = jnp.exp(eta + jnp.exp(ancillary) * ndtri(safe_q))
    elif isinstance(family, LogLogisticAFT):
        value = jnp.exp(
            eta + (jnp.log(safe_q) - jnp.log1p(-safe_q)) * jnp.exp(-ancillary)
        )
    else:
        msg = "Unsupported AFT family."
        raise TypeError(msg)
    return jnp.where((q > 0) & (q < 1) & jnp.isfinite(value), value, jnp.nan)


def survival_mean(parameters: ArrayLike, design: ArrayLike, family: AFTFamily) -> Array:
    """Unconditional mean duration; infinite or unrepresentable means are NaN."""
    eta, ancillary = _location(parameters, design)
    if isinstance(family, WeibullAFT):
        log_mean = eta + gammaln(1 + jnp.exp(-ancillary))
        exists = jnp.isfinite(ancillary)
    elif isinstance(family, LogNormalAFT):
        log_mean = eta + jnp.exp(2 * ancillary) / 2
        exists = jnp.isfinite(ancillary)
    elif isinstance(family, LogLogisticAFT):
        inverse_shape = jnp.exp(-ancillary)
        exists = jnp.isfinite(ancillary) & (ancillary > 0)
        safe_inverse = jnp.where(exists, inverse_shape, 0.5)
        delta = jnp.where(exists, -jnp.expm1(-ancillary), 0.5)
        near_one = safe_inverse > 0.5  # noqa: PLR2004
        safe_delta = jnp.where(near_one, delta, 0.5)
        near_inverse = jnp.where(near_one, safe_inverse, 0.5)
        far_inverse = jnp.where(near_one, 0.5, safe_inverse)
        log_mean = eta + jnp.where(
            near_one,
            jnp.log(near_inverse) - jnp.log(safe_delta) - jnp.log(jnp.sinc(safe_delta)),
            -jnp.log(jnp.sinc(far_inverse)),
        )
    else:
        msg = "Unsupported AFT family."
        raise TypeError(msg)
    value = jnp.exp(log_mean)
    return jnp.where(exists & jnp.isfinite(value), value, jnp.nan)


@lru_cache(maxsize=4)
def _legendre_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Construct Gauss-Legendre nodes and weights on [0, 1] on the host."""
    nodes: list[float] = []
    weights: list[float] = []
    convergence = 1e-15
    for i in range(1, order + 1):
        root = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(20):
            previous, current = 1.0, root
            for degree in range(2, order + 1):
                previous, current = (
                    current,
                    ((2 * degree - 1) * root * current - (degree - 1) * previous)
                    / degree,
                )
            derivative = order * (root * current - previous) / (root * root - 1)
            step = current / derivative
            root -= step
            if abs(step) < convergence:
                break
        nodes.append((root + 1) / 2)
        weights.append(1 / ((1 - root * root) * derivative * derivative))
    return tuple(nodes), tuple(weights)


def _loglogistic_rmst(eta: Array, ancillary: Array, horizon: Array) -> Array:
    """Compare bounded quadrature orders; agreement is not an error bound."""
    dtype = jnp.result_type(eta, ancillary, horizon, 1.0)
    shape = jnp.exp(ancillary)
    maximum = jnp.finfo(dtype).max
    scale = jnp.exp(jnp.minimum(eta, jnp.log(maximum) - 1))
    split = jnp.minimum(horizon, scale)
    # Very small shapes need no extreme concentration near u=1. An unbounded
    # power loses every quadrature node to underflow and can falsely agree at
    # zero. Bound the coordinate change and evaluate its Jacobian in log space.
    exponent = jnp.clip(jnp.ceil(2 * jnp.exp(-ancillary)), 1.0, 32.0)

    def integrate(order: int) -> Array:
        nodes, weights = _legendre_rule(order)
        u = jnp.asarray(nodes, dtype=dtype)
        w = jnp.asarray(weights, dtype=dtype)
        log_u = jnp.log(u)
        log_split = jnp.log(jnp.where(split > 0, split, 1))
        first_log_t = log_split[..., None] + exponent[..., None] * log_u
        second_t = split[..., None] + (horizon - split)[..., None] * u
        first_z = shape[..., None] * (first_log_t - eta[..., None])
        second_z = shape[..., None] * (
            jnp.log(jnp.where(second_t > 0, second_t, 1)) - eta[..., None]
        )
        first = jnp.where(
            split[..., None] > 0,
            jnp.exp(
                log_split[..., None]
                + jnp.log(exponent[..., None])
                + (exponent[..., None] - 1) * log_u
                + jax.nn.log_sigmoid(-first_z)
            ),
            0,
        )
        second = (horizon - split)[..., None]
        return jnp.sum(
            w * (first + second * jax.nn.sigmoid(-second_z)),
            axis=-1,
        )

    previous = integrate(32)
    result = jnp.full_like(previous, jnp.nan)
    tolerance = jnp.sqrt(jnp.finfo(dtype).eps)
    for order in (64, 128, 256):
        current = integrate(order)
        converged = jnp.abs(current - previous) <= tolerance * jnp.maximum(
            jnp.abs(current), jnp.finfo(dtype).tiny
        )
        lower = horizon * jax.nn.sigmoid(
            -shape * (jnp.log(jnp.where(horizon > 0, horizon, 1)) - eta)
        )
        slack = tolerance * horizon
        bounded = (current >= lower - slack) & (current <= horizon + slack)
        result = jnp.where(
            jnp.isnan(result) & converged & bounded & jnp.isfinite(current),
            current,
            result,
        )
        previous = current
    return result


def restricted_mean(  # noqa: PLR0914  # distinct closed-form branches
    parameters: ArrayLike, design: ArrayLike, horizon: ArrayLike, family: AFTFamily
) -> Array:
    """Integral of marginal survival through a finite nonnegative horizon.

    Weibull and lognormal use incomplete-gamma and normal-CDF formulas.
    Log-logistic uses bounded quadrature and returns NaN when successive orders
    do not agree to the square root of machine epsilon; this is a convergence
    diagnostic, not a certified error bound. Unrepresentable results are NaN.
    """
    eta, ancillary = _location(parameters, design)
    h = jnp.broadcast_arrays(jnp.asarray(horizon), eta)[0]
    valid_h = jnp.isfinite(h) & (h >= 0)
    log_h = jnp.log(jnp.where(valid_h & (h > 0), h, 1))
    if isinstance(family, WeibullAFT):
        inverse_shape = jnp.exp(-ancillary)
        log_power = jnp.exp(ancillary) * (log_h - eta)
        # Beyond the largest representable exponent, the regularized gamma is
        # one. Substitute a finite argument before exponentiating so reverse
        # mode does not encounter the undefined product zero times infinity.
        limit = jnp.log(jnp.finfo(jnp.result_type(log_power)).max) - 2
        saturated = log_power > limit
        power = jnp.exp(jnp.where(saturated, 0, log_power))
        regularized = jnp.where(saturated, 1, gammainc(inverse_shape, power))
        value = jnp.exp(eta + gammaln(1 + inverse_shape) + jnp.log(regularized))
        value = jnp.where(regularized > 0, value, jnp.nan)
    elif isinstance(family, LogNormalAFT):
        sigma = jnp.exp(ancillary)
        z = (log_h - eta) / sigma
        log_first = eta + sigma * sigma / 2 + log_ndtr(z - sigma)
        log_second = log_h + log_ndtr(-z)
        value = jnp.exp(jnp.logaddexp(log_first, log_second))
    elif isinstance(family, LogLogisticAFT):
        value = _loglogistic_rmst(eta, ancillary, jnp.where(valid_h, h, 0))
    else:
        message = "Unsupported AFT family."
        raise TypeError(message)
    value = jnp.where(h == 0, 0, value)
    return jnp.where(valid_h & jnp.isfinite(value), value, jnp.nan)


__all__ = [
    "AFTFamily",
    "LogLogisticAFT",
    "LogNormalAFT",
    "WeibullAFT",
    "cumulative_hazard",
    "log_density",
    "log_likelihood_rows",
    "log_survival",
    "restricted_mean",
    "survival_mean",
    "survival_probability",
    "survival_quantile",
]
