"""Response and reduced-residual permutation statistics for fixed geometry.

Schedules index response rows, jointly in both Gram axes. Freedman--Lane uses
the reduced fitted signal plus permuted reduced residuals; it remains an
approximate nuisance-adjusted procedure even when every schedule is enumerated.
Calibration and exchangeability declarations belong to the host inference door.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from statadic.association.geometric import geometric_projection

__all__ = ["GeometricNull", "freedman_lane_null", "response_permutation_null"]


class GeometricNull(NamedTuple):
    """Every attempted statistic and its numerical availability, without deletion."""

    values: jax.Array
    valid: jax.Array


@jax.jit
def response_permutation_null(
    gram: jax.Array,
    full_basis: jax.Array,
    reduced_basis: jax.Array,
    schedules: jax.Array,
) -> GeometricNull:
    """Permute response rows using one retained row-index schedule at a time."""

    def evaluate(order: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = geometric_projection(
            gram[order[:, None], order[None, :]], full_basis, reduced_basis
        )
        return result.statistic, result.valid

    values, valid = jax.lax.map(evaluate, schedules)
    return GeometricNull(values, valid)


@jax.jit
def freedman_lane_null(
    gram: jax.Array,
    full_basis: jax.Array,
    reduced_basis: jax.Array,
    schedules: jax.Array,
) -> GeometricNull:
    """Evaluate ``(H_reduced + P R_reduced) B (...)'`` without coordinates.

    Fitted/residual Gram blocks are formed once per call. ``lax.map`` retains
    scalar nulls rather
    than a permutations-by-observations-by-observations stack of transformed
    response matrices. No geometry, kernel parameter or predictor is refitted.
    """
    fitted_left = reduced_basis @ (reduced_basis.T @ gram)
    fitted = (fitted_left @ reduced_basis) @ reduced_basis.T
    cross = fitted_left - fitted
    residual = gram - fitted_left - fitted_left.T + fitted

    def evaluate(order: jax.Array) -> tuple[jax.Array, jax.Array]:
        # H B H + H B R P' + P R B H + P R B R P'. The reduced
        # projection is fixed, so each draw uses indexing and thin products.
        permuted = (
            fitted
            + cross[:, order]
            + cross.T[order, :]
            + residual[order[:, None], order[None, :]]
        )
        # Retain the exact observed array for the identity transformation:
        # recombining fitted/residual blocks can otherwise round a tied F down.
        permuted = jnp.where(
            jnp.all(order == jnp.arange(gram.shape[0])), gram, permuted
        )
        result = geometric_projection(permuted, full_basis, reduced_basis)
        return result.statistic, result.valid

    values, valid = jax.lax.map(evaluate, schedules)
    return GeometricNull(values, valid)
