# jaxtyping shape strings are runtime contracts.
"""Energy distance from a target sample to a weighted mixture of candidates.

The evaluation half of the kernel family: given weights from any source
(:mod:`statadic.discrepancy.mmd.energy.balancing.weights`, the exact barycentre in
:mod:`statadic.geometry.barycentre`, or a caller's own procedure), score them.
"""

from __future__ import annotations

import jax.numpy as jnp

from statadic.discrepancy.mmd.components import candidate_energy_components
from statadic.discrepancy.mmd.energy.balancing._common import _MAX_DISTANCE_EXPONENT
from statadic.discrepancy.mmd.energy.mixture import _mixture_energy_value
from statadic.geometry.ground.config import (  # runtime annotations  # noqa: TC001
    GroundDistanceSelection,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (
    cdist,
)
from statadic.sample.packed import PackedClouds  # runtime contract  # noqa: TC001

__all__ = [
    "energy_distance_kernel_mixture",
]


def energy_distance_kernel_mixture(
    target: PackedClouds,
    candidates: PackedClouds,
    weights: jnp.ndarray,
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    precomputed_within: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Compute energy distance from target to weighted mixture of candidates.

    Evaluates the energy distance E(target, mixture) where mixture is defined
    by the convex combination: mixture = sum_k weights[k] * candidates[k].

    This is useful for evaluating the quality of mixture weights obtained from
    kernel_balancing_weights or other optimization procedures.

    Args:
        target: Single PackedClouds distribution with atom probabilities.
        candidates: PackedClouds collection with possibly unequal true counts.
        weights: Mixture weights array of shape (K,), should sum to 1.
        metric: Distance metric (see cdist for options).
        exponent: Distance exponent α in (0, 2).
        precomputed_components: Optional ``(XX, YX)`` pair to reuse instead
            of recomputing the Gram/cross terms; see
            :func:`energy_barycentre_weights` for the bit-identical-input
            requirement.
        precomputed_within: Optional precomputed target within-term
            ``mean(cdist(target, target, metric=metric) ** exponent)`` (the
            ``YY`` term) to reuse instead of recomputing it.

    Returns:
        Energy distance between target and weighted mixture (non-negative scalar).

    Examples:
        >>> target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # 2 samples
        >>> cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])  # 3 samples
        >>> candidates = [cand1, cand2]
        >>> weights = jnp.array([1.0, 0.0])
        >>> float(
        ...     energy_distance_kernel_mixture(
        ...         PackedClouds.from_points(target),
        ...         PackedClouds.from_clouds(candidates),
        ...         weights,
        ...     )
        ... )
        0.0

    """
    if not 0.0 < exponent < _MAX_DISTANCE_EXPONENT:
        message = f"exponent must be in (0, 2), got {exponent}"
        raise ValueError(message)
    resolved_metric = resolve_ground_distance(metric)

    # Compute energy components
    if precomputed_components is not None:
        candidate_gram, target_cross = precomputed_components
    else:
        candidate_gram, target_cross = candidate_energy_components(
            target,
            candidates,
            resolved_metric,
            exponent,
        )

    # Also need YY (mean distance within target)
    if precomputed_within is not None:
        target_self = jnp.asarray(precomputed_within)
    else:

        def ground(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return cdist(a, b, metric=resolved_metric) ** exponent

        target_self = target.expectation(target, ground)

    return _mixture_energy_value(weights, target_cross, candidate_gram, target_self)
