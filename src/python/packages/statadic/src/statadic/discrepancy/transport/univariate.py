"""Exact probability transport on the real line with unequal weighted supports.

The last axis indexes atoms; leading axes broadcast. Atom weights are normalized
separately to probability masses, so this is not unbalanced transport. Zero-mass
atoms are inactive, including their coordinates. Numeric domain failures return
NaN per lane; structural misuse raises before tracing. Order is static and finite
with ``p >= 1``. Location and mass derivatives are piecewise, away from changes of
support ordering and cumulative-mass order. Rooting adds a singularity at zero;
use the powered objective when that distinction matters for learning.
"""

from __future__ import annotations

import math
from functools import partial
from numbers import Real
from typing import TYPE_CHECKING, Final

import jax
import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001

if TYPE_CHECKING:
    from statadic.sample.packed import PackedClouds

__all__ = ["wasserstein_1d", "wasserstein_1d_power"]

_SQUARED_ORDER: Final = 2.0


def _order(p: object) -> float:
    if isinstance(p, bool) or not isinstance(p, Real):
        message = "Wasserstein order must be a finite real number >= 1."
        raise TypeError(message)
    value = float(p)
    if not math.isfinite(value) or value < 1:
        message = "Wasserstein order must be a finite real number >= 1."
        raise ValueError(message)
    return value


@static_jaxtyping("n")
def _quantiles(
    sample: PackedClouds,
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Sort sites and their masses together and close the cumulative law at one.

    The law closes at the last positive-mass atom, so trailing inactive atoms keep
    zero-width intervals instead of receiving the cumulative-sum roundoff.
    """
    values, weights = jax.lax.sort_key_val(
        sample.coordinates[:, 0] + 0.0, sample.masses, is_stable=True
    )
    total = jnp.sum(weights)
    cdf = jnp.cumsum(weights / jnp.where(total > 0, total, 1))
    index = jnp.arange(weights.size)
    last = jnp.max(jnp.where(weights > 0, index, -1))
    last = jnp.where(last < 0, weights.size - 1, last)
    return values, jnp.where(index >= last, 1, jnp.minimum(cdf, 1))


@static_jaxtyping("n m")
def _quantile_cost(x: PackedClouds, y: PackedClouds, *, p: float) -> Array:
    left, left_cdf = _quantiles(x)
    right, right_cdf = _quantiles(y)
    # A merged endpoint advances the quantile index on its own side. A
    # prefix sum thus locates both quantiles of every interval without two
    # binary searches per interval or rounding an interior midpoint.
    from_left = jnp.arange(left.size + right.size) < left.size
    ends, sides = jax.lax.sort_key_val(
        jnp.concatenate((left_cdf, right_cdf)), from_left, is_stable=True
    )
    starts = jnp.concatenate((jnp.zeros_like(ends[:1]), ends[:-1]))
    i = jnp.cumsum(sides, dtype=jnp.int32) - sides.astype(jnp.int32)
    j = jnp.arange(ends.size) - i
    active = ends > starts
    # Repeated CDF endpoints have no transport mass. Mask sites before their
    # difference/power: an unused extreme gap can otherwise overflow and make
    # zero times infinity corrupt even an exact self-distance.
    gap = jnp.where(active, left[jnp.minimum(i, left.size - 1)], 0) - jnp.where(
        active, right[jnp.minimum(j, right.size - 1)], 0
    )
    costs = jnp.square(gap) if p == _SQUARED_ORDER else jnp.abs(gap) ** p
    return jnp.where(x.valid & y.valid, jnp.sum((ends - starts) * costs), jnp.nan)


def _cdf_distance(x: PackedClouds, y: PackedClouds) -> Array:
    """W1 is the area between the empirical CDFs on the pooled support."""
    sites = jnp.concatenate((x.coordinates[:, 0] + 0.0, y.coordinates[:, 0] + 0.0))
    weights = jnp.concatenate(
        (x.masses / jnp.sum(x.masses), -y.masses / jnp.sum(y.masses))
    )
    # Park padding after the genuine sites and exclude its intervals. Moving a
    # zero-mass atom must not introduce a long interval of roundoff-sized CDF.
    locations, mass_changes = jax.lax.sort_key_val(
        jnp.where(weights != 0, sites, jnp.inf), weights, is_stable=True
    )
    delta = jnp.cumsum(mass_changes)[:-1]
    active = jnp.isfinite(locations[1:]) & (delta != 0)
    gaps = jnp.where(active, locations[1:], 0) - jnp.where(active, locations[:-1], 0)
    return jnp.where(x.valid & y.valid, jnp.sum(gaps * jnp.abs(delta)), jnp.nan)


@partial(jax.jit, static_argnames=("p", "rooted"))
def _transport(x: PackedClouds, y: PackedClouds, *, p: float, rooted: bool) -> Array:
    """Compile validation, quantile integration and optional rooting together."""
    if x.data.shape[-1] != 1 or y.data.shape[-1] != 1:
        msg = "One-dimensional transport requires one feature per atom"
        raise ValueError(msg)
    if not x.data.shape[-2] or not y.data.shape[-2]:
        msg = "Transport requires nonempty last-axis atom supports."
        raise ValueError(msg)
    if p == 1:
        return x.map_aligned(y, _cdf_distance)
    cost = x.map_aligned(y, partial(_quantile_cost, p=p))
    if not rooted:
        return cost
    return jnp.sqrt(cost) if p == _SQUARED_ORDER else cost ** (1.0 / p)


@static_jaxtyping("*batch n m")
def wasserstein_1d_power(
    x: PackedClouds,
    y: PackedClouds,
    *,
    p: int | float = 2,  # beartype's runtime numeric tower  # noqa: PYI041
) -> Float[Array, "*batch"]:
    """Return exact ``W_p**p`` for broadcast weighted real-line measures.

    Args:
        x: First locations, shape ``(*batch, n)``.
        y: Second locations, shape ``(*batch, m)``; ``m`` may differ from ``n``.
        x_weights: Broadcastable atom weights; omitted means uniform masses.
        y_weights: Broadcastable atom weights for the second support.
        p: Static, finite order at least one.

    Returns:
        Powered probability-transport costs over the broadcast batch shape.
        Invalid numeric lanes return NaN, including zero or negative total mass.

    Raises:
        TypeError: If coordinates or weights are complex, or order is not a real number.
        ValueError: If order, atom axes, weight shapes or batch shapes are invalid.

    """
    return _transport(x, y, p=_order(p), rooted=False)


@static_jaxtyping("*batch")
def wasserstein_1d(
    x: PackedClouds,
    y: PackedClouds,
    *,
    p: int | float = 2,  # beartype's runtime numeric tower  # noqa: PYI041
) -> Float[Array, "*batch"]:
    """Return rooted ``W_p`` under the same contract as :func:`wasserstein_1d_power`.

    Rooted distance need not be differentiable at zero. The powered operation
    provides the unrooted objective explicitly rather than changing its meaning.

    """
    return _transport(x, y, p=_order(p), rooted=True)
