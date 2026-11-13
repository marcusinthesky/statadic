# jaxtyping shape strings are runtime contracts.
"""Energy-functional matrices and raw empirical ground means.

Cloud adapters preserve true counts, V-statistic self-pairs and supplied order.
Array reductions remain separate from law-bearing construction: the typed
energy-functional door retains its geometry and estimator provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.axioms import (  # runtime contract  # noqa: TC001
    Law,
    PremetricLaw,
)
from statadic.core.domains import (  # runtime annotations  # noqa: TC001
    DirectOrigin,
    EmpiricalDistribution,
)
from statadic.core.matrices import (  # noqa: TC001  # evaluated runtime annotations
    DMat,
    EnergyFunctional,
    MatrixProvenance,  # runtime annotations
    PairwiseEstimateMatrix,
    VStatisticScheme,
    _dmat,
    _pairwise_estimate,
)
from statadic.core.statistics import (  # noqa: TC001
    VStatisticEstimator,  # runtime annotations
)
from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.mmd.energy.geometry import (
    parse_distribution_exponent,
    resolve_energy_geometry,
)
from statadic.discrepancy.mmd.energy.scalar import (
    _energy_distance_core,
    _resolve_distribution_inputs,
)
from statadic.discrepancy.provenance import (  # noqa: TC001
    ExpectedGroundDistance,  # runtime annotations
)
from statadic.geometry.ground.config import (  # runtime annotations  # noqa: TC001
    GroundDistanceSelection,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (  # runtime annotations  # noqa: TC001
    EUCLIDEAN,
    GroundDistance,
    cdist,
)

if TYPE_CHECKING:
    from statadic.sample.packed import (
        PackedClouds,
    )

__all__ = [
    "empirical_mean_ground_distances",
    "energy_distance_matrix",
    "energy_functional_matrix",
    "mean_distance_estimate_matrix",
    "mean_distance_matrix",
]


@static_jaxtyping("sets")
def energy_distance_matrix(
    sample_sets: PackedClouds,
    exponent: float = 1.0,
    *,
    metric: GroundDistanceSelection,
) -> Float[Array, "sets sets"]:
    """Symmetric matrix of pairwise energy distances between samples.

    Native pytree batching evaluates the scalar mass-weighted functional for
    each pair at fixed capacities. Inactive storage carries no probability.
    Upper-triangle values are mirrored for exact symmetry. Prepare ragged
    external supports through the checked factories before tracing.

    Args:
        sample_sets: Nonempty PackedClouds collection; true atom counts may differ.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        Symmetric ``(n_sets, n_sets)`` matrix ``D`` with ``D[i, j]`` the energy
        distance between ``sample_sets[i]`` and ``sample_sets[j]`` and an
        exactly-zero diagonal.

    Raises:
        ValueError: If exponent is not in (0, 2), if the list is empty, if a
            cloud is not rank two or has zero rows, or if feature widths differ.
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    Examples:
        >>> from statadic.sample.packed import PackedClouds
        >>> a = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> b = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        >>> clouds = PackedClouds.from_clouds([a, b])
        >>> d = energy_distance_matrix(clouds, exponent=1.0, metric="euclidean")
        >>> d.shape
        (2, 2)
        >>> bool(d[0, 1] == d[1, 0]) and bool(d[0, 0] == 0.0)
        True

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        return _energy_distance_core(a, b, validated_exponent.value, resolved_metric)

    values = sample_sets.map_cross(sample_sets, pair)
    upper = jnp.triu(values, k=1)
    symmetric = upper + upper.T
    diagonal = jnp.diag(values)
    return symmetric.at[jnp.diag_indices(sample_sets.n_observations)].set(
        jnp.where(jnp.isfinite(diagonal), 0, diagonal)
    )


@static_jaxtyping("m d sets")
def mean_distance_matrix(
    sample_sets: PackedClouds,
    exponent: float = 1.0,
    *,
    metric: GroundDistanceSelection,
) -> Float[Array, "sets sets"]:
    """Full pairwise mean-distance matrix, including diagonal, over packed samples.

    Unlike :func:`energy_distance_matrix` (which combines terms into the
    energy-distance statistic and zeroes the diagonal), this returns the raw
    per-pair mean distance ``M[i, j] = mean(cdist(a[i], a[j]) ** exponent)``
    for every ``(i, j)`` including ``i == j``.

    This producer retains energy-ground theorem validation. Use
    :func:`empirical_mean_ground_distances` for a typed raw estimate with no
    energy-distance claim.

    Args:
        sample_sets: Packed clouds with shared feature width.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        ``(n_sets, n_sets)`` matrix ``M``, see above.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    return _weighted_ground_means(
        sample_sets, resolved_metric, validated_exponent.value
    )


def _weighted_ground_means(
    clouds: PackedClouds, metric: GroundDistance[Law], exponent: float
) -> Array:
    """Reduce point costs against empirical probabilities, including self-pairs."""

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        active = (a.masses[:, None] > 0) & (b.masses[None, :] > 0)
        costs = (
            jnp.where(active, cdist(a.coordinates, b.coordinates, metric=metric), 0)
            ** exponent
        )
        return jnp.where(
            a.valid & b.valid,
            jnp.sum(costs * a.masses[:, None] * b.masses[None, :]),
            jnp.nan,
        )

    return clouds.map_cross(clouds, pair)


def empirical_mean_ground_distances(
    sample_sets: PackedClouds,
    exponent: float = 1.0,
    metric: GroundDistanceSelection = EUCLIDEAN,
) -> PairwiseEstimateMatrix[
    EmpiricalDistribution,
    ExpectedGroundDistance,
    VStatisticEstimator,
    DirectOrigin,
]:
    """Estimate pairwise expected powered ground distances between empirical clouds.

    Entry ``(i, j)`` is the explicit mean of ``d(x, y) ** exponent`` over all
    ``x`` in cloud ``i`` and ``y`` in cloud ``j``. Self-pairs are included:
    the diagonal records within-cloud dispersion and generally is nonzero.
    These raw estimates carry no metric or energy-distance claim. In
    particular, powering angular or cosine grounds above one requires no
    energy-distance theorem for this operation.

    Prepared clouds have fixed capacities with dynamic true counts and atom
    masses. Padded rows have neither observations nor mass. Ragged preparation
    stays eager; the probability reductions support JAX transformations.

    Args:
        sample_sets: Nonempty PackedClouds collection with shared feature width.
        exponent: Numeric power in the open interval ``(0, 2)``.
        metric: Declared ground strategy or closed built-in selection. Its
            value-domain requirements, such as nonzero angular rows, apply.

    Returns:
        Pairwise empirical V-statistic estimates, including within-cloud means.

    Raises:
        ValueError: If the exponent is outside ``(0, 2)``, the list is empty,
            a cloud is not rank two or has no rows, or feature widths differ.

    """
    validated_exponent = parse_distribution_exponent(exponent)
    resolved_metric = resolve_ground_distance(metric)
    values = _weighted_ground_means(
        sample_sets, resolved_metric, validated_exponent.value
    )
    return _pairwise_estimate(values)


def energy_functional_matrix(
    sample_sets: PackedClouds,
    exponent: float = 1.0,
    *,
    metric: GroundDistanceSelection,
) -> DMat[
    EmpiricalDistribution,
    MatrixProvenance[PremetricLaw, EnergyFunctional[VStatisticScheme]],
]:
    """Brand :func:`energy_distance_matrix` with its domain, law, and origin.

    The typed successor to the unbranded producer: same numbers, same ragged
    admission, and same masked count arithmetic, with the provenance the array
    form cannot carry. Only this form can enter
    :func:`statadic.core.matrices.metrize_energy`.

    What the three parameters claim, and what they deliberately do not:

    * ``EmpiricalDistribution`` — the compared objects are finitely-supported
      empirical measures. Never ``PopulationLaw``: a sample is not a law.
    * ``PremetricLaw`` — nonnegative and symmetric with a zero diagonal, and
      **not** separating. The energy functional separates laws only over a
      compatible separating ground/exponent construction. Its actual resolved
      ground, exponent and result axioms are retained as static evidence, so
      ``metrize_energy`` cannot substitute an unrelated ground hypothesis.
      In particular, the ``angular`` ground cannot see scale at all.
    * ``EnergyFunctional[VStatisticScheme]`` — the biased plug-in ``S``, not the
      population functional and not its root. This is the origin
      :func:`metrize_energy` demands, so a transport or checked-door matrix with
      identical law capabilities is a static error there.

    Args:
        sample_sets: Clouds admitted exactly as :func:`energy_distance_matrix`
            admits them; row counts may differ, feature widths may not.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        The branded energy functional matrix, ready for
        :func:`statadic.core.matrices.metrize_energy`.

    Raises:
        ValueError: Whatever :func:`energy_distance_matrix` raises.

    Examples:
        >>> from statadic.sample.packed import PackedClouds
        >>> a = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> b = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        >>> energy_functional_matrix(
        ...     PackedClouds.from_clouds([a, b]), metric="euclidean"
        ... ).values.shape
        (2, 2)

    """
    geometry = resolve_energy_geometry(metric, exponent)
    construction = EnergyFunctional[VStatisticScheme](
        geometry.ground.name, geometry.exponent.value, geometry.law.axioms
    )
    return _dmat(
        energy_distance_matrix(
            sample_sets, geometry.exponent.value, metric=geometry.ground
        ),
        construction=construction,
    )


@static_jaxtyping("m d")
def mean_distance_estimate_matrix(
    sample_sets: PackedClouds,
    exponent: float = 1.0,
    *,
    metric: GroundDistanceSelection,
) -> PairwiseEstimateMatrix[
    EmpiricalDistribution,
    ExpectedGroundDistance,
    VStatisticEstimator,
    DirectOrigin,
]:
    """Brand :func:`mean_distance_matrix` as a realized estimate, not a distance.

    A ``PairwiseEstimateMatrix`` rather than a ``DMat``, and that is the whole
    content of the signature: ``M[i, i] = E[d(X, X')^α]`` is the within-cloud
    dispersion, which does not vanish, so no zero-diagonal, identity, or
    metrization law applies. Branding it a dissimilarity would let it flow into
    a consumer that assumes one.

    ``DirectOrigin``, not ``PoweredGroundOrigin``: the exponent is a runtime
    float that may be exactly one, so the wider origin is the claim this
    producer can actually support.

    Args:
        sample_sets: Packed clouds admitted by :func:`mean_distance_matrix`.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        The branded pairwise mean-distance estimates, diagonal included.

    Raises:
        ValueError: Whatever :func:`mean_distance_matrix` raises.

    """
    return _pairwise_estimate(
        mean_distance_matrix(sample_sets, exponent, metric=metric)
    )
