# jaxtyping shape strings are runtime contracts.
"""Scalar energy-distance estimators and their shared exponent contract.

The theorem/configuration root is :mod:`statadic.discrepancy.mmd.energy.geometry`.
This scalar layer depends on it and owns the shared eager resolver consumed by
matrix, statistic, and mixture siblings, keeping the family dependency graph
acyclic.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import jax.numpy as jnp
from jax import jit, lax

from statadic.core.axioms import (
    SEMIMETRIC,
    Law,
    Premetric,
    Semimetric,
)
from statadic.core.typing import static_jaxtyping

if TYPE_CHECKING:
    from statadic.core.axioms import (
        Axioms,
    )
    from statadic.sample.packed import PackedClouds

from statadic.core.typing import Array  # noqa: TC001
from statadic.discrepancy.mmd.energy.geometry import (  # runtime annotations  # noqa: TC001
    UNIT_EXPONENT,
    DistributionExponent,
    EnergyGeometry,
    _DeclaredEnergyGeometry,
    _energy_result_law,
    parse_distribution_exponent,
)
from statadic.geometry.ground.config import (  # runtime annotations  # noqa: TC001
    GroundDistanceSelection,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (  # runtime annotations  # noqa: TC001
    ANGULAR,
    EUCLIDEAN,
    GroundDistance,
    cdist,
)

__all__ = [
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "IncompatibleEnergyGroundExponentError",
    "energy_distance",
    "energy_distance_with_geometry",
]

_DECLARED_ENERGY_GEOMETRIES: Final = {
    "energy_distance[euclidean]": (EUCLIDEAN, UNIT_EXPONENT),
    "energy_distance[angular]": (ANGULAR, UNIT_EXPONENT),
}

DECLARED_AXIOMS: Final[dict[str, Axioms]] = {
    name: _energy_result_law(ground.law, ground.name, exponent).axioms
    for name, (ground, exponent) in _DECLARED_ENERGY_GEOMETRIES.items()
}
"""Derived property-battery view; energy geometry laws are authoritative."""

DECLARED_BRANDS: Final[dict[str, type[Premetric]]] = {
    name: Semimetric if axioms & SEMIMETRIC == SEMIMETRIC else Premetric
    for name, axioms in DECLARED_AXIOMS.items()
}
"""Derived migration-only marker view; numerical producers never consume it."""


class IncompatibleEnergyGroundExponentError(ValueError):
    """Raised when a valid exponent lacks the selected ground-law theorem."""

    def __init__(self, metric: str, exponent: object) -> None:
        """Name both parts of the incompatible eager configuration."""
        super().__init__(
            f"ground distance {metric!r} is incompatible with "
            f"energy exponent {exponent!r}"
        )


def _resolve_distribution_inputs(
    exponent: float,
    metric: GroundDistanceSelection,
) -> tuple[DistributionExponent, GroundDistance[Law]]:
    """Parse an eager compatibility selection and validate its theorem branch."""
    validated_exponent = parse_distribution_exponent(exponent)
    resolved_metric = resolve_ground_distance(metric)
    try:
        _energy_result_law(
            resolved_metric.law,
            resolved_metric.name,
            validated_exponent,
        )
    except TypeError:
        raise IncompatibleEnergyGroundExponentError(
            resolved_metric.name,
            exponent,
        ) from None
    return validated_exponent, resolved_metric


# Above this logical coordinate-difference size, stream rows on CPU. Small
# and accelerator workloads retain the dense, parallel reduction.
_DENSE_DIFFERENCE_BYTES: Final = 1 << 20


@partial(jit, static_argnums=(2,))
@static_jaxtyping("n d m")
def _energy_distance_core(
    x: PackedClouds,
    y: PackedClouds,
    exponent: float,
    metric: GroundDistance[Law],
) -> Array:
    """Fused scalar energy-distance core (JIT-compiled).

    Computes the three ``cdist`` calls (``xy``, ``xx``, ``yy``), applies the
    exponent, and reduces to the scalar energy-distance statistic inside a
    single jitted trace so XLA can fuse the shared norm/matmul work rather
    than materializing three eager intermediates.

    Args:
        x: First sample, shape (n, d).
        y: Second sample, shape (m, d).
        exponent: Distance exponent (static).
        metric: Declared ground-distance strategy with optional dynamic leaves.

    Returns:
        Energy distance value.

    """
    left, right = x.coordinates, y.coordinates

    def block(a: Array, b: Array, wa: Array, wb: Array) -> Array:
        active = (wa[:, None] > 0) & (wb[None, :] > 0)
        distances = jnp.where(active, cdist(a, b, metric=metric), 0)
        zero = distances == 0
        # Self-pairs are identically zero even for fractional exponents. Guard
        # the power's argument so its inactive infinite derivative cannot
        # contaminate the coordinate derivative of that constant self-pair.
        distances = jnp.where(zero, 0.0, jnp.where(zero, 1.0, distances) ** exponent)
        return jnp.sum(distances * wa[:, None] * wb[None, :])

    def stream(a: Array, b: Array, wa: Array, wb: Array) -> Array:
        def row(pair: tuple[Array, Array]) -> Array:
            site, mass = pair
            return block(site[None, :], b, mass[None], wb)

        return jnp.sum(lax.map(row, (a, wa)))

    def expectation(a: Array, b: Array, wa: Array, wb: Array) -> Array:
        size = a.size * b.shape[0] * jnp.result_type(a, b).itemsize
        if metric is EUCLIDEAN and size > _DENSE_DIFFERENCE_BYTES:
            return lax.platform_dependent(a, b, wa, wb, cpu=stream, default=block)
        return block(a, b, wa, wb)

    cross = expectation(left, right, x.masses, y.masses)
    within_x = expectation(left, left, x.masses, x.masses)
    within_y = expectation(right, right, y.masses, y.masses)
    return jnp.where(x.valid & y.valid, 2 * cross - within_x - within_y, jnp.nan)


@static_jaxtyping("n d m")
def energy_distance(
    x: PackedClouds,
    y: PackedClouds,
    exponent: float = 1.0,
    *,
    metric: GroundDistanceSelection,
) -> Array:
    """Compute energy distance between two samples.

    Energy distance is a statistical distance between probability distributions
    based on expected distances between random samples. For samples X and Y:

    E(X,Y) = 2·E[||X-Y||^α] - E[||X-X'||^α] - E[||Y-Y'||^α]

    where X' and Y' are independent copies and α is the distance exponent.

    Axioms: see :data:`DECLARED_AXIOMS` — ``SEMIMETRIC`` under
    ``metric="euclidean"``, only ``NONNEGATIVE | SYMMETRY`` under the
    ``metric="angular"``, which cannot see scale. Typed metrization
    is available only through :func:`statadic.core.matrices.metrize_energy`, whose
    finite empirical V-statistic construction retains the ground and exponent
    provenance required for the declared law.

    Notes:
        For the finite-support energy and kernel identity, see
        :ref:`formal-STATADIC-ENERGY-001` and
        :cite:p:`sejdinovic-equivalence-2013`. Its kernel assumptions and exact
        arithmetic do not verify this floating-point implementation.

    Args:
        x: First ``PackedClouds`` distribution, including atom probabilities.
        y: Second distribution; leading cloud axes broadcast against ``x``.
        exponent: Distance exponent α. Must be in (0, 2) for valid metric.
            The default is 1.0; the distinct α=2 ANOVA regime is represented by
            :class:`AnovaExponent`, not accepted by this distribution distance.
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        Raw V-energy array over broadcast cloud axes; scalar for two single
        distributions. Roundoff can produce a small negative value.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    Examples:
        >>> import jax.numpy as jnp
        >>> from statadic.sample.packed import PackedClouds
        >>> from statadic.discrepancy.mmd.energy import energy_distance
        >>> x = PackedClouds.from_points(jnp.array([[0.0], [1.0]]))
        >>> y = PackedClouds.from_points(jnp.array([[2.0], [3.0]]))
        >>> float(energy_distance(x, y, exponent=1.0, metric="euclidean"))
        3.0

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        return _energy_distance_core(a, b, validated_exponent.value, resolved_metric)

    return x.map_aligned(y, pair)


@static_jaxtyping("n d m")
def energy_distance_with_geometry[
    GroundLawT: Law,
    ExponentT: DistributionExponent,
    EnergyLawT: Law,
](
    x: PackedClouds,
    y: PackedClouds,
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT],
) -> Array:
    """Evaluate a prevalidated distribution-sensitive energy geometry.

    Args:
        x: First ``PackedClouds`` distribution with atom probabilities.
        y: Second distribution, with broadcast-compatible leading cloud axes.
        geometry: Compatible typed ground/exponent/result-law construction.

    Returns:
        Raw JAX energy functional over broadcast cloud axes.

    """
    if not isinstance(geometry, _DeclaredEnergyGeometry):
        message = (
            "geometry must be constructed by energy_geometry or unsafe_energy_geometry"
        )
        raise TypeError(message)

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        return _energy_distance_core(a, b, geometry.exponent.value, geometry.ground)

    return x.map_aligned(y, pair)
