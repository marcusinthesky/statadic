"""Balanced entropic OT on finite probability measures, with log-domain updates."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from statadic.core.typing import Array  # noqa: TC001
from statadic.geometry.ground.metrics import squared_euclidean_distance
from statadic.sample.packed import PackedClouds  # noqa: TC001

__all__ = ["sinkhorn_cost"]


def sinkhorn_cost(
    left: PackedClouds,
    right: PackedClouds,
    *,
    epsilon: float,
    iterations: int,
    tolerance: float,
) -> Array:
    """Dual objective with KL relative to the product of the supplied atom masses.

    Fixed-count log-domain updates support reverse-mode AD. A failed marginal
    residual returns NaN rather than exposing an unconverged transport cost.
    """
    a, b = left.masses, right.masses
    cost = squared_euclidean_distance(left.coordinates, right.coordinates)
    log_a = jnp.where(a > 0, jnp.log(jnp.where(a > 0, a, 1)), -jnp.inf)
    log_b = jnp.where(b > 0, jnp.log(jnp.where(b > 0, b, 1)), -jnp.inf)

    def update(_: Array, potentials: tuple[Array, Array]) -> tuple[Array, Array]:
        _, g = potentials
        f = -epsilon * logsumexp(log_b + (g - cost) / epsilon, axis=1)
        g = -epsilon * logsumexp(log_a[:, None] + (f[:, None] - cost) / epsilon, axis=0)
        return f, g

    f, g = jax.lax.fori_loop(
        0, iterations, update, (jnp.zeros_like(a), jnp.zeros_like(b))
    )
    plan = jnp.exp(log_a[:, None] + log_b + (f[:, None] + g - cost) / epsilon)
    error = jnp.maximum(
        jnp.max(jnp.abs(plan.sum(axis=1) - a)), jnp.max(jnp.abs(plan.sum(axis=0) - b))
    )
    dual = a @ f + b @ g - epsilon * (plan.sum() - 1)
    return jnp.where(left.valid & right.valid & (error <= tolerance), dual, jnp.nan)
