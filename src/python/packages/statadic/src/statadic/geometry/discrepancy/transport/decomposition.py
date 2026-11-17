"""Location and centered-shape diagnostics for Euclidean Wasserstein geometry.

The squared W2 decomposition is specific to Euclidean ground distance. Its
location component is the squared distance between empirical means; its shape
component compares the two centered empirical measures. Both transport values
are computed independently, so their reconstruction error is inspectable.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from statadic.core.typing import Array  # noqa: TC001
from statadic.geometry.discrepancy.transport.empirical import wasserstein_distance_jax
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.measure.clouds import PackedClouds  # noqa: TC001

__all__ = [
    "EuclideanWasserstein2Decomposition",
    "euclidean_wasserstein2_decomposition",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EuclideanWasserstein2Decomposition:
    """Squared Euclidean W2 split into location and centered shape.

    ``reconstruction_error`` is the signed residual ``distance_squared -
    location_squared - shape_squared``. The shape term comes from a separate
    transport solve, rather than subtraction. ``finite`` records whether every
    reported component is finite; it does not independently certify either
    transport optimum. Squaring can overflow even for a finite distance.
    """

    distance_squared: Array
    location_squared: Array
    shape_squared: Array
    reconstruction_error: Array
    finite: Array


def euclidean_wasserstein2_decomposition(
    left: PackedClouds,
    right: PackedClouds,
    *,
    max_iterations: int = 10000,
    tolerance: float = 1e-6,
) -> EuclideanWasserstein2Decomposition:
    """Separate squared Euclidean W2 into location and centered-shape terms.

    Supports the same empirical measures as ``wasserstein_distance_jax``,
    including unequal supports and nonuniform probabilities. Uniform equal-size
    inputs retain assignment specialization. Weighted-solver iteration and
    numerical tolerances apply identically to original and centered problems.
    Outputs broadcast over cloud axes and remain JIT/vmap-transformable.
    """
    left_mean, right_mean = left.mean(), right.mean()
    left_centered = left.with_coordinates(left.coordinates - left_mean[..., None, :])
    right_centered = right.with_coordinates(
        right.coordinates - right_mean[..., None, :]
    )
    total = jnp.square(
        wasserstein_distance_jax(
            left,
            right,
            2.0,
            ground=EUCLIDEAN,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    )
    shape = jnp.square(
        wasserstein_distance_jax(
            left_centered,
            right_centered,
            2.0,
            ground=EUCLIDEAN,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    )
    location = jnp.sum(jnp.square(left_mean - right_mean), axis=-1)
    error = total - (location + shape)
    finite = (
        jnp.isfinite(total)
        & jnp.isfinite(location)
        & jnp.isfinite(shape)
        & jnp.isfinite(error)
    )
    return EuclideanWasserstein2Decomposition(total, location, shape, error, finite)
