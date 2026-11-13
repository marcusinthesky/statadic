# jaxtyping shape strings are runtime contracts.
"""Raw MMD² estimate matrices and branded V-statistic distance matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.mmd.contracts import _require_v_distance
from statadic.discrepancy.mmd.scalar import _shared_reference, mmd_squared
from statadic.geometry.ground.similarities import (
    LINEAR_KERNEL,
    SimilarityStrategy,
)
from statadic.sample.packed import (  # runtime annotations  # noqa: TC001
    PackedClouds,
)

__all__ = ["mmd_matrix_values", "mmd_squared_matrix"]

if TYPE_CHECKING:
    from statadic.geometry.ground.similarities import PositiveSemidefiniteKernelLaw

_PACKED_DIMENSIONS = 3


def mmd_squared_matrix(
    sample_sets: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: bool = False,
) -> Float[Array, "sets sets"]:
    """Construct an unbranded symmetric matrix of pairwise MMD² estimates.

    Unlike :func:`mmd_matrix`, this function permits ``unbiased=True`` and
    returns the signed U-statistic estimates without a square root, clamp, or
    ``DMat`` brand.  The U-statistic diagonal need not be zero.

    Args:
        sample_sets: Nonempty PackedClouds collection with shared feature width.
            True support counts may differ; atom masses are probabilities.
        kernel: Declared PSD strategy, optionally carrying dynamic parameters.
        reference: Reference shared by every distance-induced Gram block.
        unbiased: Drop the within-sample diagonals.

    Returns:
        Symmetric raw-estimate matrix of shape ``(sets, sets)``.

    Raises:
        InsufficientUStatisticSampleError: If a U-statistic cloud is singleton.
        TypeError: If a bare string or callable bypasses declaration.
        ValueError: If the list is empty, a cloud is not rank two, the feature
            widths differ, or any cloud has zero rows.

    Notes:
        Normalizing kernels and grounds require the evidence named by their
        strategy. This transformable array door leaves an out-of-domain value
        visible as ``NaN``; checked carrier doors reject eagerly.

        Native pytree batching evaluates the same scalar probability functional
        for every pair. Prepare ragged supports eagerly before tracing; fixed
        capacities and count/mass leaves remain dynamic. U-statistics require
        uniform atom masses and at least two true atoms per distribution.

    """
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    if sample_sets.data.ndim != _PACKED_DIMENSIONS or not sample_sets.n_observations:
        message = "MMD matrices require a nonempty collection of clouds"
        raise ValueError(message)
    shared = _shared_reference(
        sample_sets.coordinates.reshape(-1, sample_sets.data.shape[-1]),
        reference,
        sample_sets.masses.reshape(-1),
    )

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        return mmd_squared(a, b, kernel=kernel, reference=shared, unbiased=unbiased)

    values = sample_sets.map_cross(sample_sets, pair)
    if not unbiased:
        diagonal = jnp.diag(values)
        values = values.at[jnp.diag_indices(sample_sets.n_observations)].set(
            jnp.where(jnp.isfinite(diagonal), 0, diagonal)
        )
    return values


@static_jaxtyping("d sets")
def mmd_matrix_values(
    sample_sets: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: Literal[False] = False,
) -> Float[Array, "sets sets"]:
    """Construct the raw rooted V-statistic MMD value matrix.

    Args:
        sample_sets: Nonempty list of rank-two ``(m_k, d)`` clouds, admitted on
            exactly the terms :func:`mmd_squared_matrix` admits: row counts
            ``m_k`` may differ, the feature width ``d`` may not, and no cloud
            may have zero rows.
        kernel: Declared PSD strategy, optionally carrying dynamic parameters.
        reference: Reference shared by every distance-induced Gram block.
        unbiased: Must remain ``False``; retained to reject old calls clearly.

    Returns:
        Symmetric rooted MMD values with a zero diagonal. Semantic brands are
        attached only by :func:`rooted_mmd_matrix`.

    Raises:
        UnbiasedMmdDistanceError: If ``unbiased=True`` is supplied at runtime.
        ValueError: If :func:`mmd_squared_matrix` rejects the cloud list.

    """
    _require_v_distance(unbiased=unbiased)
    squared: Float[Array, "sets sets"] = mmd_squared_matrix(
        sample_sets,
        kernel=kernel,
        reference=reference,
        unbiased=False,
    )
    clamped = jnp.maximum(squared, 0.0)
    values: Float[Array, "sets sets"] = jnp.sqrt(clamped)
    diagonal = jnp.diag(values)
    return values.at[jnp.diag_indices(values.shape[0])].set(
        jnp.where(jnp.isfinite(diagonal), 0, diagonal)
    )
