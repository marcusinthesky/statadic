# jaxtyping shape strings are runtime contracts.
"""Kernel-balancing simplex weights and their leave-one-out driver.

The stationarity-residual surrogate is ``min_w ||XX w - YX||^2``. The exact
finite-mixture energy barycentre lives in :mod:`statadic.geometry.barycentre`.
"""

from __future__ import annotations

import jax.numpy as jnp

from statadic.discrepancy.mmd.components import candidate_energy_components
from statadic.discrepancy.mmd.energy.balancing._common import _MAX_DISTANCE_EXPONENT
from statadic.geometry.ground.config import (  # runtime annotations  # noqa: TC001
    GroundDistanceSelection,
    resolve_ground_distance,
)
from statadic.optimize.simplex import pgd_simplex_affine
from statadic.sample.packed import PackedClouds, pack

__all__ = [
    "energy_distance_kernel_pairwise",
    "kernel_balancing_weights",
]


def kernel_balancing_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    solver: str = "pgd",
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> jnp.ndarray:
    """Find simplex weights by the kernel-balancing surrogate.

    Solves the quadratic program:

        min_n  ||XX @ n - YX||²
        s.t.   sum(n) = 1
               0 <= n <= 1

    where XX[i,j] = E[||Xi - Xj||^α] is the mean pairwise distance between
    candidates i and j, and YX[k] = E[||Y - Xk||^α] is the mean distance
    from target Y to candidate Xk.

    This objective is a stationarity-residual surrogate. It is not the finite-
    mixture energy objective minimized by :func:`energy_barycentre_weights`.

    Args:
        target: Target distribution array of shape (n, d).
        candidates: Either:
            - Array of shape (K, m, d) with uniform sizes (backward compatible)
            - List of K arrays with shapes (m_k, d) for irregular sizes (recommended)
        metric: Distance metric (see cdist for options).
        exponent: Distance exponent α in (0, 2). Default 1.0.
        solver: Projected gradient descent on the simplex ("pgd").
        tol: Solver tolerance for convergence.
        maxiter: Maximum solver iterations.

    Returns:
        Array of shape (K,) containing optimal mixture weights.
        Weights sum to 1 and lie in [0, 1].

    Raises:
        ValueError: If solver is not recognized or exponent not in (0, 2).

    Examples:
        >>> # Irregular sizes (recommended)
        >>> target = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        >>> cand1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # 2 samples
        >>> cand2 = jnp.array([[10.0, 10.0], [10.1, 10.1], [10.2, 10.2]])  # 3 samples
        >>> candidates = [cand1, cand2]  # List of arrays
        >>> weights = kernel_balancing_weights(target, candidates)
        >>> bool(jnp.allclose(weights, jnp.array([1.0, 0.0])))
        True

        >>> # Uniform sizes (backward compatible)
        >>> cand1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        >>> cand2 = jnp.array([[10.0, 10.0], [10.1, 10.1]])
        >>> candidates = jnp.stack([cand1, cand2])  # Shape (2, 2, 2)
        >>> weights = kernel_balancing_weights(target, candidates)

    """
    if not 0.0 < exponent < _MAX_DISTANCE_EXPONENT:
        message = f"exponent must be in (0, 2), got {exponent}"
        raise ValueError(message)
    resolved_metric = resolve_ground_distance(metric)

    # Convert uniform array to list if needed (backward compatibility)
    if isinstance(candidates, jnp.ndarray):
        candidates_list = [candidates[i] for i in range(candidates.shape[0])]
    else:
        candidates_list = candidates

    # Compute energy distance components
    candidate_gram, target_cross = candidate_energy_components(
        PackedClouds.from_array(target),
        pack(candidates_list),
        resolved_metric,
        exponent,
    )

    # Define the quadratic objective: min ||XX @ n - YX||²
    # This expands to: min n^T (XX^T XX) n - 2 (XX^T YX)^T n + YX^T YX
    # In standard form: min 1/2 n^T Q n + c^T n
    # where Q = 2 * XX^T XX and c = -2 * XX^T YX

    quadratic = 2 * (candidate_gram.T @ candidate_gram)
    c = -2 * (candidate_gram.T @ target_cross)

    if solver != "pgd":
        message = f"Unknown solver: {solver}. Use 'pgd'."
        raise ValueError(message)
    return pgd_simplex_affine(quadratic, c, maxiter=maxiter, tol=tol)


def energy_distance_kernel_pairwise(
    distributions: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    solver: str = "pgd",
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> jnp.ndarray:
    """Compute optimal mixture weights for all pairwise comparisons.

    For each distribution i, finds mixture weights for all other distributions
    {j : j != i} that minimize energy distance to distribution i.

    Args:
        distributions: Either:
            - Array of shape (K, m, d) with uniform sizes (backward compatible)
            - List of K arrays with shapes (m_k, d) for irregular sizes (recommended)
        metric: Distance metric (see cdist for options).
        exponent: Distance exponent α in (0, 2).
        solver: Projected gradient descent on the simplex ("pgd").
        tol: Solver tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        Array of shape (K, K-1) where row i contains optimal weights
        for approximating distribution i using all others.

    Examples:
        >>> # Irregular sizes
        >>> dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # 2 samples
        >>> dist2 = jnp.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])  # 3 samples
        >>> dist3 = jnp.array([[10.0, 10.0]])  # 1 sample
        >>> distributions = [dist1, dist2, dist3]
        >>> weights = energy_distance_kernel_pairwise(distributions)
        >>> weights.shape
        (3, 2)

        >>> # Uniform sizes (backward compatible)
        >>> dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        >>> dist2 = jnp.array([[1.0, 1.0], [1.1, 1.1]])
        >>> dist3 = jnp.array([[10.0, 10.0], [10.1, 10.1]])
        >>> distributions = jnp.stack([dist1, dist2, dist3])
        >>> weights = energy_distance_kernel_pairwise(distributions)
        >>> weights.shape
        (3, 2)

    """
    # Convert uniform array to list if needed
    if isinstance(distributions, jnp.ndarray):
        distributions_list = [distributions[i] for i in range(distributions.shape[0])]
    else:
        distributions_list = distributions

    distribution_count = len(distributions_list)
    resolved_metric = resolve_ground_distance(metric)

    # Compute weights for each target
    weights_list = []
    for i in range(distribution_count):
        target = distributions_list[i]

        # Build candidate list excluding index i
        candidates = [
            distributions_list[j] for j in range(distribution_count) if j != i
        ]

        w = kernel_balancing_weights(
            target, candidates, resolved_metric, exponent, solver, tol, maxiter
        )
        weights_list.append(w)

    return jnp.stack(weights_list, axis=0)
