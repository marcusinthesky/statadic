# jaxtyping shape strings are runtime contracts.
"""Uncalibrated two-sample statistics from energy discrepancies.

Use sample clouds, distance matrices or precomputed block reductions as supplied
by the corresponding door. These functions return numerical statistics;
permutation schedules and test calibration belong to ``statadic.inference``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.mmd.energy.scalar import (
    _energy_distance_core,
    _resolve_distribution_inputs,
)
from statadic.geometry.ground.config import (  # noqa: TC001
    GroundDistanceSelection,  # runtime annotations
)
from statadic.sample.packed import PackedClouds

__all__ = [
    "energy_test_statistic",
    "energy_test_statistic_from_blocks",
    "energy_test_statistic_from_distances",
]


@static_jaxtyping("n d m")
def energy_test_statistic(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    exponent: float = 1.0,
    *,
    metric: GroundDistanceSelection,
) -> Float[Array, ""]:
    """Compute energy test statistic for two-sample test.

    The test statistic is the energy distance scaled by sample sizes:
    T(X,Y) = (n·m)/(n+m) · E(X,Y)

    This scaling makes the statistic comparable across different sample sizes
    and provides better power properties for hypothesis testing.

    Args:
        x: First sample, array of shape (n, d).
        y: Second sample, array of shape (m, d).
        exponent: Distance exponent. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        Zero-dimensional JAX array containing the non-negative test statistic.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    Examples:
        >>> import jax.numpy as jnp
        >>> from statadic.discrepancy.mmd.energy import energy_test_statistic
        >>> x = jnp.array([[0.0], [1.0]])
        >>> y = jnp.array([[2.0], [3.0]])
        >>> float(energy_test_statistic(x, y, metric="euclidean"))
        3.0

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    n, m = len(x), len(y)
    ed: Float[Array, ""] = _energy_distance_core(
        PackedClouds.from_array(x),
        PackedClouds.from_array(y),
        validated_exponent.value,
        resolved_metric,
    )
    return (n * m) / (n + m) * ed


@static_jaxtyping("total")
def energy_test_statistic_from_distances(
    distances: Float[Array, "total total"], n: int
) -> Float[Array, ""]:
    """Energy test statistic from a precomputed, pre-exponentiated pooled matrix.

    Distance-hoist companion to :func:`energy_test_statistic` (t05.9): under a
    label permutation of a two-sample problem, the *pooled* (n+m, n+m)
    pairwise distance matrix ``D = cdist(concat(x, y), concat(x, y)) ** α`` is
    invariant — only the row/column order (which points fall in the X block
    vs the Y block) changes. This lets a caller compute ``D`` **once**, then
    reconstruct the statistic for every permutation replicate by slicing a
    reindexed view of ``D`` (``D[perm][:, perm]``) rather than recomputing
    ``cdist`` from scratch each time (see the permutation stage's distance
    hoist, t05.2).

    Algebraically identical to ``energy_test_statistic(x, y, exponent, metric=metric)``
    when ``distances`` is exactly ``cdist(concat(x, y), concat(x, y),
    metric=metric) ** exponent`` and ``x`` occupies the first ``n`` rows/cols.

    Args:
        distances: Pooled (n+m, n+m) distance matrix, already raised to the
            exponent α. Rows/columns ordered ``[X (n), Y (m)]``.
        n: Size of the first group. A Python ``int``, not a tracer: it is a
            slice bound, so it must be static.

    Returns:
        Scalar energy test statistic ``T(X, Y) = (n·m)/(n+m) · E(X, Y)``.

    """
    total = distances.shape[0]
    m = total - n
    dxy = distances[:n, n:]
    dxx = distances[:n, :n]
    dyy = distances[n:, n:]
    term1 = 2.0 * jnp.sum(dxy) / (n * m)
    term2 = jnp.sum(dxx) / (n * n)
    term3 = jnp.sum(dyy) / (m * m)
    ed = term1 - term2 - term3
    return (n * m) / (n + m) * ed


def energy_test_statistic_from_blocks(
    block_sums: Float[Array, "2 2"],  # jaxtyping fixed shape
    n: int,
    m: int,
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Energy test statistic from two-sample group-block distance sums.

    Block-sum companion to :func:`energy_test_statistic_from_distances`. That
    function slices a pooled matrix whose rows are already ordered ``[X, Y]``;
    this one consumes the reduction ``B = C D Cᵀ`` directly, so a permutation
    replicate never materialises its own reindexed ``D`` (see
    :mod:`statadic.inference.resampling.blocks` for the identity and the
    measurement that motivates it).

    ``block_sums[0, 0] = Σ_{i,j∈X} d``, ``block_sums[0, 1] = Σ_{i∈X, j∈Y} d``
    and ``block_sums[1, 1] = Σ_{i,j∈Y} d``, so the three terms below are the
    same ``dxy``/``dxx``/``dyy`` sums, in the same order and with the same
    divisors, as the sliced form — only the summation order inside each block
    differs (one GEMM accumulation rather than a ``jnp.sum`` tree reduction).

    Args:
        block_sums: Symmetric ``(2, 2)`` group-block sums ordered ``[X, Y]``.
        n: Size of the first group.
        m: Size of the second group.

    Returns:
        Scalar energy test statistic ``T(X, Y) = (n·m)/(n+m) · E(X, Y)``.

    """
    term1 = 2.0 * block_sums[0, 1] / (n * m)
    term2 = block_sums[0, 0] / (n * n)
    term3 = block_sums[1, 1] / (m * m)
    ed = term1 - term2 - term3
    return (n * m) / (n + m) * ed
