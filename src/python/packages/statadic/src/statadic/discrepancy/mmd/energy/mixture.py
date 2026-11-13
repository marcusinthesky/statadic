# jaxtyping shape strings are runtime contracts.
"""Energy discrepancies for fixed mixtures of candidate distributions.

Sample-cloud and grouped-block reductions evaluate the same V-statistic
objective. They select no weights, resampling scheme or reference distribution;
balancing, barycentre fitting and inferential calibration have separate owners.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.axioms import (
    Law,  # runtime contract  # noqa: TC001
)
from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.mmd.components import candidate_energy_components
from statadic.discrepancy.mmd.energy.scalar import _resolve_distribution_inputs
from statadic.geometry.ground.config import (  # noqa: TC001
    GroundDistanceSelection,  # runtime annotations
)
from statadic.geometry.ground.metrics import (  # runtime annotations  # noqa: TC001
    GroundDistance,
    cdist,
)
from statadic.sample.packed import PackedClouds  # runtime contract  # noqa: TC001

__all__ = ["mixture_energy_distance", "mixture_energy_distance_from_blocks"]


def _mixture_energy_value(
    weights: jnp.ndarray,
    cross: jnp.ndarray,
    gram: jnp.ndarray,
    target_self: jnp.ndarray,
) -> jnp.ndarray:
    """Reduce retained V-statistic components without selecting any weights."""
    return 2 * jnp.dot(cross, weights) - jnp.dot(weights, gram @ weights) - target_self


@static_jaxtyping("n d k")
def _energy_terms(
    x: PackedClouds,
    candidates: PackedClouds,
    exponent: float,
    metric: GroundDistance[Law],
) -> tuple[Float[Array, ""], Float[Array, " k"], Float[Array, "k k"]]:
    """Return target and candidate energy V-statistic components.

    Args:
        x: Target sample, shape (n, d).
        candidates: K candidate samples.
        exponent: Distance exponent α.
        metric: Declared ground-distance strategy.

    Returns:
        Tuple (term_xx scalar, target_cross shape (K,), candidate_gram shape (K, K)).

    """

    def ground(a: Array, b: Array) -> Array:
        return cdist(a, b, metric=metric) ** exponent

    term_xx = x.expectation(x, ground)
    candidate_gram, target_cross = candidate_energy_components(
        x, candidates, metric, exponent
    )
    return term_xx, target_cross, candidate_gram


@static_jaxtyping("n d k")
def mixture_energy_distance(
    x: PackedClouds,
    candidates: PackedClouds,
    weights: Float[Array, " k"],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
) -> Float[Array, ""]:
    """Energy distance ``E(X, Y_w)`` between target and weighted mixture.

    ``E = 2 wᵀ·XY - E[d(x,x')] - wᵀ·candidate_gram·w`` (nonnegative by negative type for
    ``metric``-induced ground distance with ``exponent ∈ (0,2)``).

    Args:
        x: Target empirical distribution with atom probabilities.
        candidates: Collection of candidate empirical distributions.
        weights: Mixture weights, shape (K,).
        exponent: Distance exponent α in (0, 2).
        metric: Distance metric.

    Returns:
        Scalar energy distance ``E(X, Y_w)``.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    terms: tuple[Float[Array, ""], Float[Array, " k"], Float[Array, "k k"]] = (
        _energy_terms(
            x,
            candidates,
            validated_exponent.value,
            resolved_metric,
        )
    )
    term_xx, target_cross, candidate_gram = terms
    return _mixture_energy_value(weights, target_cross, candidate_gram, term_xx)


def mixture_energy_distance_from_blocks(
    block_sums: jnp.ndarray, sizes: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Mixture energy V-statistic from pooled group-block distance sums.

    ``block_sums[a, b] = Σ_{i∈a, j∈b} d(i, j)`` over the ``(K+1)`` pooled groups
    (group 0 is the target ``X``); dividing by ``sizes[a]·sizes[b]`` recovers the
    block means, so ``E = 2 wᵀ·XY - term_xx - wᵀ·YY·w`` exactly as in
    :func:`statadic.discrepancy.mmd.energy.mixture_energy_distance` (self-distances
    contribute
    zero and are kept in the diagonal counts, matching the V-statistic
    normalization).

    Returns:
        Scalar mixture energy distance ``E(X, Y_w)``.

    """
    block_means = block_sums / (sizes[:, None] * sizes[None, :])
    return _mixture_energy_value(
        weights, block_means[0, 1:], block_means[1:, 1:], block_means[0, 0]
    )
