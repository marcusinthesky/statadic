# jaxtyping shape strings are runtime contracts.
"""Scalar MMD estimates and the firm-level cosine-mean comparator."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal  # runtime contract

import jax.numpy as jnp
from jax import jit
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.mmd.contracts import (
    _require_u_sample_size,
    _require_v_distance,
)
from statadic.geometry.ground.similarities import (
    LINEAR_KERNEL,
    SimilarityStrategy,
    gram,
)

if TYPE_CHECKING:
    from statadic.geometry.ground.similarities import PositiveSemidefiniteKernelLaw
    from statadic.sample.packed import PackedClouds

__all__ = ["cosine_mean_mmd", "mmd", "mmd_squared"]

_U_MINIMUM = 2


@static_jaxtyping("n d")
def _shared_reference(
    anchor: Float[Array, "n d"],
    reference: Float[Array, " d"] | None,
    masses: Array | None = None,
) -> Float[Array, " d"]:
    """Resolve the single reference shared by every Gram block.

    Distance-induced reference terms cancel from MMD only when ``Kxx``, ``Kyy``
    and ``Kxy`` use the same point. Select a finite positive-mass support point,
    preserving the first atom for ordinary uniform clouds. Angular grounds
    require every positive-mass support point to be nonzero.
    """
    if reference is not None:
        return reference
    eligible = jnp.all(jnp.isfinite(anchor), axis=-1)
    if masses is not None:
        eligible &= masses > 0
    return anchor[jnp.argmax(eligible)]


@partial(
    jit,
    static_argnames=("unbiased",),
)
@static_jaxtyping("n d m")
def _mmd_squared_core(  # noqa: PLR0914
    x: PackedClouds,
    y: PackedClouds,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: Float[Array, " d"] | None,
    *,
    unbiased: bool,
) -> Array:
    """Evaluate the V- or U-statistic MMD² estimate in one compiled core."""
    left, right = x.coordinates, y.coordinates
    if kernel is LINEAR_KERNEL:
        dtype = jnp.result_type(
            left, right, reference if reference is not None else 1.0
        )
        left, right = left.astype(dtype), right.astype(dtype)
        if reference is not None:
            reference = reference.astype(dtype)
    shared_reference = _shared_reference(left, reference, x.masses)
    n, m = x.counts, y.counts
    kxx = gram(left, left, kernel, reference=shared_reference)
    kyy = gram(right, right, kernel, reference=shared_reference)
    kxy = gram(left, right, kernel, reference=shared_reference)
    wx, wy = x.masses, y.masses

    def expectation(values: Array, a: Array, b: Array) -> Array:
        active = (a[:, None] > 0) & (b[None, :] > 0)
        return jnp.sum(jnp.where(active, values, 0) * a[:, None] * b[None, :])

    valid = x.valid & y.valid
    if unbiased:
        valid &= (n >= _U_MINIMUM) & (m >= _U_MINIMUM) & x.is_uniform & y.is_uniform
        xx = (
            (
                expectation(kxx, wx, wx)
                - jnp.sum(jnp.where(wx > 0, jnp.diag(kxx), 0) * wx**2)
            )
            * n
            / (n - 1)
        )
        yy = (
            (
                expectation(kyy, wy, wy)
                - jnp.sum(jnp.where(wy > 0, jnp.diag(kyy), 0) * wy**2)
            )
            * m
            / (m - 1)
        )
    else:
        if kernel is LINEAR_KERNEL:
            return jnp.where(
                valid,
                jnp.sum(
                    (
                        jnp.sum((left - shared_reference) * wx[:, None], axis=0)
                        - jnp.sum((right - shared_reference) * wy[:, None], axis=0)
                    )
                    ** 2
                ),
                jnp.nan,
            )
        xx = expectation(kxx, wx, wx)
        yy = expectation(kyy, wy, wy)
    # A PSD kernel is symmetric. Average both floating-point orientations so
    # every declared strategy realizes that law without name-based dispatch.
    kyx = gram(right, left, kernel, reference=shared_reference)
    cross = 0.5 * (expectation(kxy, wx, wy) + expectation(kyx, wy, wx))
    return jnp.where(valid, xx + yy - 2.0 * cross, jnp.nan)


@static_jaxtyping("n d m")
def mmd_squared(
    x: PackedClouds,
    y: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: bool = False,
) -> Array:
    """Compute a raw squared maximum-mean-discrepancy estimate.

    ``unbiased=False`` is the V-statistic and is the only convention represented
    in ``DECLARED_AXIOMS``/``DECLARED_BRANDS``.  ``unbiased=True`` returns the
    unbranded U-statistic estimate unchanged: it may be negative for distinct
    finite samples and is never clamped.

    Args:
        x: First ``PackedClouds`` distribution with atom probabilities.
        y: Second distribution, with broadcast-compatible leading cloud axes.
        kernel: Declared PSD strategy. Numerical parameter leaves may be dynamic;
            its mathematical law and input domain remain fixed.
        reference: Shared distance-induced reference point.
        unbiased: Drop within-sample diagonals to form the U-statistic.

    Returns:
        MMD² over broadcast cloud axes, scalar for two single distributions.
        A U-statistic result may be negative.

    Raises:
        InsufficientUStatisticSampleError: If either U-statistic block has fewer
            than two observations.
        TypeError: If a bare string or callable bypasses declaration.

    Notes:
        Normalizing kernels and grounds require the evidence named by their
        strategy. This numerical carrier door leaves an out-of-domain value
        visible as ``NaN``; checked factories reject eagerly.

    """
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    if unbiased:
        _require_u_sample_size(x.data.shape[-2])
        _require_u_sample_size(y.data.shape[-2])

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        return _mmd_squared_core(a, b, kernel, reference, unbiased=unbiased)

    return x.map_aligned(y, pair)


@static_jaxtyping("n d m")
def mmd(
    x: PackedClouds,
    y: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: Literal[False] = False,
) -> Array:
    """Compute the V-statistic MMD distance between two empirical measures.

    The U-statistic estimates MMD² and can be negative, so ``unbiased=True`` is
    rejected here both statically and at runtime.  Use :func:`mmd_squared` for
    that unbranded estimate.

    Args:
        x: First ``PackedClouds`` distribution with atom probabilities.
        y: Second distribution, with broadcast-compatible leading cloud axes.
        kernel: Declared PSD strategy, optionally carrying dynamic parameters.
        reference: Shared distance-induced reference point.
        unbiased: Must remain ``False``; retained to reject old calls clearly.

    Returns:
        Nonnegative square root of the V-statistic MMD² value.

    Raises:
        UnbiasedMmdDistanceError: If ``unbiased=True`` is supplied at runtime.

    """
    _require_v_distance(unbiased=unbiased)
    squared: Float[Array, ""] = mmd_squared(
        x,
        y,
        kernel=kernel,
        reference=reference,
        unbiased=False,
    )
    clamped = jnp.maximum(squared, 0.0)
    return jnp.sqrt(clamped)


@jit
@static_jaxtyping("d")
def cosine_mean_mmd(
    psi_x: Float[Array, " d"],
    psi_y: Float[Array, " d"],
) -> Array:
    """Compare two already-normalized prototype directions by chord distance.

    This is an MMD between point masses on the space of empirical measures, not
    a ``kernel=`` choice for article-cloud :func:`mmd`. Pooling and
    normalization are pipeline transformations; this core does neither.

    Args:
        psi_x: First pooled mean embedding, shape ``(d,)``.
        psi_y: Second pooled mean embedding, shape ``(d,)``.

    Returns:
        Chord distance between the supplied unit directions.

    """
    return jnp.linalg.norm(psi_x - psi_y)
