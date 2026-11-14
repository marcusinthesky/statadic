# jaxtyping shape strings are runtime contracts.
"""Spatial IV response maps and quadratic-moment variability.

These component operations are shared by spatial GMM fitting and post-fit
inference. Inputs must already describe the same complete, identified sample;
they do not validate instrument identification or authorize a reference law.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from statadic.core.typing import (
    Array,  # numerical annotations  # noqa: TC001
)

__all__ = ["iv_response_map", "spatial_moment_variability"]


def iv_response_map(design: Array, instruments: Array) -> tuple[Array, Array]:
    """Return a stable 2SLS response map and first-stage rank."""
    qh, _ = jnp.linalg.qr(instruments, mode="reduced")
    projected = qh @ (qh.T @ design)
    q, r = jnp.linalg.qr(projected, mode="reduced")
    return solve_triangular(r, q.T, lower=False), jnp.linalg.matrix_rank(projected)


def spatial_moment_variability(
    residual: Array,
    lam: Array,
    w: Array,
    matrices: Array,
    design: Array,
    instruments: Array,
    *,
    heteroskedastic: bool,
    endogenous: bool,
) -> tuple[Array, Array, Array]:
    """Retain quadratic variability and endogenous-coefficient correction vectors."""
    n = design.shape[0]
    filtered_design = design - lam * (w @ design)
    filtered = residual - lam * (w @ residual)
    beta_map, _ = iv_response_map(
        filtered_design, instruments if endogenous else filtered_design
    )
    adjustments = jnp.zeros((n, 2), dtype=w.dtype)
    if endogenous:
        derivatives = (
            -2 * jnp.einsum("ip,aji,j->pa", filtered_design, matrices, filtered) / n
        )
        adjustments = n * beta_map.T @ derivatives
    if heteroskedastic:
        sigma = filtered**2
        variability = (
            2 * jnp.einsum("aij,bji,i,j->ab", matrices, matrices, sigma, sigma) / n
        )
        variability += adjustments.T @ (sigma[:, None] * adjustments) / n
    else:
        sigma2 = jnp.mean(filtered**2)
        mu3 = jnp.mean(filtered**3)
        excess = jnp.mean(filtered**4) - 3 * sigma2**2
        diagonal = jnp.diagonal(matrices, axis1=1, axis2=2).T
        variability = 2 * sigma2**2 * jnp.einsum("aij,bji->ab", matrices, matrices) / n
        variability += excess * (diagonal.T @ diagonal) / n
        variability += sigma2 * (adjustments.T @ adjustments) / n
        variability += mu3 * (adjustments.T @ diagonal + diagonal.T @ adjustments) / n
    return variability, adjustments, beta_map
