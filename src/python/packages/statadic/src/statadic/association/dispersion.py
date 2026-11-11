# jaxtyping shape strings are runtime contracts.
r"""DISCO: the energy-distance decomposition of total dispersion.

DISCO decomposes total pairwise dispersion using a distance matrix and a group
assignment. PERMDISP summarizes distances to spatial medians in an embedding.
Both return descriptive statistics; neither generates a reference distribution.

The statistic
-------------
Rizzo & Székely (2010), "DISCO analysis: a nonparametric extension of analysis
of variance" — the energy-statistics-native analogue of PERMANOVA. For groups
:math:`g = 1..K` with sizes :math:`n_g`, :math:`N = \\sum_g n_g`, and pairwise
distances :math:`d_{ij}`, let :math:`\bar a_{XY}` be the mean of :math:`d_{ij}`
over :math:`i \\in X, j \\in Y`. Within-group dispersion is
:math:`S_W = \\sum_g (n_g/2)\\,\bar a_{gg}`, total dispersion is
:math:`T = (N/2)\\,\bar a`, and :math:`S_B = T - S_W`. Reported as the DISCO
index :math:`R^2_E = S_B / T` and the F-ratio
:math:`F = [S_B/(K-1)]\\,/\\,[S_W/(N-K)]`.

The ``g_alpha(A, A)`` above is a **V-statistic**: the zero diagonal is included
and the ordered sum is divided by :math:`n_g^2`, not :math:`n_g(n_g-1)`.

The PERMDISP companion consumes coordinates and summarizes distances to group
spatial medians from :mod:`statadic.geometry.median` (Anderson 2006). Its
full-model residual-permutation null lives in :mod:`statadic.inference.dispersion`.

This module owns the pure statistic kernels. DISCO label resampling and
add-one calibration live in :mod:`statadic.inference.disco`; keeping them here
would let an association-layer result bypass the hypothesis and calibration
axes of the typed inference API.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from statadic.geometry.median import weiszfeld_medians

_DISTANCE_EPSILON = 1e-12

__all__ = [
    "disco_decomposition",
    "dispersion_f_ratio",
    "dispersion_radii",
    "radius_anova_f",
]


@partial(jax.jit, static_argnames=("n_groups",))
def disco_decomposition(
    dist: jnp.ndarray, group_codes: jnp.ndarray, n_groups: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Decompose total dispersion into within- and between-group parts.

    Args:
        dist: Square distance matrix, shape ``(n, n)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.

    Returns:
        ``(S_W, S_B, T, R2_E, F)`` as JAX scalars.

    """
    n = dist.shape[0]

    def group_stats(g: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mask = (group_codes == g).astype(dist.dtype)
        n_g = jnp.sum(mask)
        # The cited DISCO g_alpha(A, A) is a V-statistic: its zero diagonal is
        # included and the ordered sum is divided by n_g^2, not n_g(n_g-1).
        sum_gg = mask @ dist @ mask
        count_gg = n_g * n_g
        abar_gg = jnp.where(
            count_gg > 0, sum_gg / jnp.where(count_gg > 0, count_gg, 1.0), 0.0
        )
        return n_g, abar_gg

    n_g_arr, abar_gg_arr = jax.vmap(group_stats)(jnp.arange(n_groups))
    s_w = jnp.sum(n_g_arr / 2.0 * abar_gg_arr)

    n_total = jnp.asarray(n, dtype=dist.dtype)
    sum_all = jnp.sum(dist)
    count_all = n_total * n_total
    abar_all = sum_all / count_all
    t_total = (n_total / 2.0) * abar_all

    s_b = t_total - s_w
    k = jnp.asarray(n_groups, dtype=dist.dtype)
    f_ratio = (s_b / (k - 1.0)) / (s_w / (n_total - k))
    r2_e = s_b / t_total
    return s_w, s_b, t_total, r2_e, f_ratio


@partial(jax.jit, static_argnames=("n_groups",))
def dispersion_f_ratio(
    coords: jnp.ndarray, group_codes: jnp.ndarray, n_groups: int
) -> jnp.ndarray:
    """Compute the PERMDISP-style homogeneity-of-dispersion F-ratio.

    Distance from each point to its own group's spatial median, summarised by
    a one-way-ANOVA F-ratio across groups (Anderson's PERMDISP, with the
    spatial median as the centroid).

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.

    Returns:
        The F-ratio as a JAX scalar.

    """
    medians = weiszfeld_medians(coords, group_codes, n_groups)
    per_point_median = medians[group_codes]
    dist_to_med = jnp.sqrt(
        jnp.sum((coords - per_point_median) ** 2, axis=1) + _DISTANCE_EPSILON
    )
    return radius_anova_f(dist_to_med, group_codes, n_groups)


@partial(jax.jit, static_argnames=("bias_adjust",))
def dispersion_radii(
    coords: jnp.ndarray,
    group_codes: jnp.ndarray,
    medians: jnp.ndarray,
    *,
    bias_adjust: bool = False,
) -> jnp.ndarray:
    """Return each point's distance to its own group's supplied centre.

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        medians: Group centres, shape ``(n_groups, k)``, for example from
            :func:`statadic.geometry.median.spatial_medians`.
        bias_adjust: Multiply radii by ``sqrt(n_g / (n_g - 1))``, vegan's
            ``betadisper(bias.adjust = TRUE)`` small-group correction. Every
            group then needs at least two points.

    Returns:
        Radii, shape ``(n,)``.

    """
    radii = jnp.linalg.norm(coords - medians[group_codes], axis=1)
    if not bias_adjust:
        return radii
    sizes = jnp.bincount(group_codes, length=medians.shape[0]).astype(radii.dtype)
    point_sizes = sizes[group_codes]
    return radii * jnp.sqrt(point_sizes / (point_sizes - 1))


@partial(jax.jit, static_argnames=("n_groups",))
def radius_anova_f(
    radii: jnp.ndarray, group_codes: jnp.ndarray, n_groups: int
) -> jnp.ndarray:
    """Summarize scalar radii by a one-way-ANOVA F-ratio across groups.

    Args:
        radii: One scalar per row, shape ``(n,)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.

    Returns:
        The F-ratio as a JAX scalar.

    """

    def group_stats(g: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mask = (group_codes == g).astype(radii.dtype)
        n_g = jnp.sum(mask)
        mean_g = jnp.sum(radii * mask) / n_g
        ss_g = jnp.sum(mask * (radii - mean_g) ** 2)
        return n_g, mean_g, ss_g

    n_g_arr, mean_g_arr, ss_within_arr = jax.vmap(group_stats)(jnp.arange(n_groups))
    grand_mean = jnp.mean(radii)
    ss_between = jnp.sum(n_g_arr * (mean_g_arr - grand_mean) ** 2)
    ss_within = jnp.sum(ss_within_arr)
    n = radii.shape[0]
    return (ss_between / (n_groups - 1)) / (ss_within / (n - n_groups))
