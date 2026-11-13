# jaxtyping shape strings are runtime contracts.
"""Full-model residual-permutation calibration for the PERMDISP statistic.

Group centres and radii are fitted once. The radii are regressed on group, the
full-model residuals are permuted, and every permuted residual vector is refit
to the same one-way design (vegan ``permutest.betadisper``; Anderson 2006).
Relabeling coordinates and refitting group medians instead is anti-conservative
when groups differ in location: mixed relabeled groups have inflated radii, so
a pure location shift is read as a dispersion difference.

The statistic is :func:`statadic.association.dispersion.radius_anova_f`; spatial
locations belong to :mod:`statadic.geometry.median`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from statadic.association.dispersion import radius_anova_f
from statadic.core.random import cell_key, permutation_batch
from statadic.core.typing import (
    ArrayLike,  # noqa: TC001
)
from statadic.geometry.median import weiszfeld_medians

__all__ = [
    "dispersion_permutation_key",
    "dispersion_permutation_null",
    "dispersion_residual_null",
]

_DISTANCE_EPSILON = 1e-12


@partial(jax.jit, static_argnames=("n_groups",))
def dispersion_residual_null(
    radii: jnp.ndarray,
    group_codes: jnp.ndarray,
    n_groups: int,
    schedules: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the residual-permutation null of :func:`radius_anova_f`.

    A draw is ``radius_anova_f(e[schedule], group_codes)`` where ``e`` are the
    full-model residuals ``radii - group_mean(radii)``. Residuals that vanish
    to roundoff carry no reference distribution, so every draw is then NaN.

    Args:
        radii: Observed radii, shape ``(n,)``, from
            :func:`statadic.association.dispersion.dispersion_radii`.
        group_codes: Observed contiguous group codes, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.
        schedules: Complete index permutations of the residuals, shape
            ``(n_permutations, n)``.

    Returns:
        Null F-ratios, shape ``(n_permutations,)``.

    """
    sizes = jnp.bincount(group_codes, length=n_groups).astype(radii.dtype)
    means = jax.ops.segment_sum(radii, group_codes, num_segments=n_groups) / sizes
    residuals = radii - means[group_codes]
    eps = jnp.finfo(radii.dtype).eps
    degenerate = jnp.linalg.norm(residuals) <= 64 * eps * jnp.linalg.norm(radii)
    draws = jax.vmap(radius_anova_f, in_axes=(0, None, None))(
        residuals[schedules], group_codes, n_groups
    )
    return jnp.where(degenerate, jnp.nan, draws)


@partial(jax.jit, static_argnames=("n_groups", "n_permutations"))
def _dispersion_permutation_null_kernel(
    coords: jnp.ndarray,
    group_codes: jnp.ndarray,
    key: jax.Array,
    n_groups: int,
    n_permutations: int,
) -> jnp.ndarray:
    """Fit the statistic's radii once and permute their residuals."""
    medians = weiszfeld_medians(coords, group_codes, n_groups)
    radii = jnp.sqrt(
        jnp.sum((coords - medians[group_codes]) ** 2, axis=1) + _DISTANCE_EPSILON
    )
    indices = jnp.arange(group_codes.shape[0], dtype=jnp.int32)
    schedules = permutation_batch(key, indices, n_permutations)
    return dispersion_residual_null(radii, group_codes, n_groups, schedules)


def dispersion_permutation_key(random_state: int) -> jax.Array:
    """Return the key owning the residual-permutation stream for ``random_state``.

    :func:`dispersion_permutation_null` permutes ``arange(n)`` index schedules
    drawn by ``permutation_batch`` under this key.
    """
    return cell_key(random_state, "geometry", "dispersion_permutation_null")


def dispersion_permutation_null(
    coords: ArrayLike,
    group_codes: ArrayLike,
    n_groups: int,
    n_permutations: int,
    random_state: int,
) -> jax.Array:
    """Draw the residual-permutation null of ``dispersion_f_ratio``.

    Radii use the same fixed-budget medians as
    :func:`statadic.association.dispersion.dispersion_f_ratio`, so the observed
    statistic and its null share one fit. Medians are not refit per draw.

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Observed group codes, shape ``(n,)``.
        n_groups: Number of distinct groups.
        n_permutations: Number of residual permutations to draw.
        random_state: Base seed for :func:`statadic.core.random.cell_key`.

    Returns:
        Permuted F-ratios, shape ``(n_permutations,)``.

    """
    if n_permutations <= 0:
        message = "n_permutations must be positive."
        raise ValueError(message)
    key = dispersion_permutation_key(random_state)
    return _dispersion_permutation_null_kernel(
        jnp.asarray(coords),
        jnp.asarray(group_codes),
        key,
        n_groups,
        n_permutations,
    )
