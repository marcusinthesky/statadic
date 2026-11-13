"""Nested-design sums of squares for one fixed response Gram matrix.

The response functional and its mathematical warrant belong to the producer.
These kernels neither truncate a spectrum nor calibrate a hypothesis test.
Design bases must already be full-rank, orthonormal and nested; host workflows
validate those preconditions before calling this fixed-shape numerical owner.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["GeometricProjection", "centered_response_gram", "geometric_projection"]


class GeometricProjection(NamedTuple):
    """Observed variation and numerical availability for one nested hypothesis."""

    within: jax.Array
    between: jax.Array
    total: jax.Array
    r_squared: jax.Array
    statistic: jax.Array
    df_test: jax.Array
    df_residual: jax.Array
    valid: jax.Array


@jax.jit
def centered_response_gram(functional: jax.Array) -> jax.Array:
    """Return minus half the double-centered supplied distance functional."""
    row_mean = jnp.mean(functional, axis=1, keepdims=True)
    column_mean = jnp.mean(functional, axis=0, keepdims=True)
    return -0.5 * (functional - row_mean - column_mean + jnp.mean(functional))


@jax.jit
def geometric_projection(
    gram: jax.Array, full_basis: jax.Array, reduced_basis: jax.Array
) -> GeometricProjection:
    """Compute a partial pseudo-F without constructing or truncating coordinates.

    ``full_basis`` and ``reduced_basis`` span the actual fitted designs. Their
    column counts determine ranks and degrees of freedom; padding an unidentified
    design with spurious QR columns violates this function's preconditions.
    """
    full = jnp.sum(full_basis * (gram @ full_basis))
    reduced = jnp.sum(reduced_basis * (gram @ reduced_basis))
    total = jnp.trace(gram)
    within = total - full
    between = full - reduced
    df_test = jnp.asarray(full_basis.shape[1] - reduced_basis.shape[1], dtype=jnp.int32)
    df_residual = jnp.asarray(gram.shape[0] - full_basis.shape[1], dtype=jnp.int32)
    statistic = (between / df_test) / (within / df_residual)
    r_squared = between / (total - reduced)
    scale = jnp.max(jnp.abs(gram))
    roundoff = 16 * gram.shape[0] * jnp.finfo(gram.dtype).eps * scale
    valid = (
        jnp.all(jnp.isfinite(gram))
        & jnp.all(jnp.isfinite(full_basis))
        & jnp.all(jnp.isfinite(reduced_basis))
        & (df_test > 0)
        & (df_residual > 0)
        & (within > roundoff)
        & (between >= -roundoff)
        & jnp.isfinite(r_squared)
        & jnp.isfinite(statistic)
    )
    return GeometricProjection(
        within, between, total, r_squared, statistic, df_test, df_residual, valid
    )
