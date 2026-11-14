# jaxtyping shape strings are runtime contracts.
"""Checked host conversion for persisted energy-functional values.

Position
--------
Consumes persisted discrepancy values and produces checked array values.

The typed JAX metrization is :func:`statadic.core.matrices.metrize_energy`. Its
signature retains domain, estimator scheme, energy origin, strong-negative-type
ground evidence, and finite-moment evidence. This module deliberately does not
offer the former ``DMat[Semimetric] -> DMat[NegativeType]`` shortcut: an
arbitrary semimetric does not carry the hypotheses needed to mint that result.

:func:`sqrt_energy_functional` remains a checked artifact adapter. It validates
sign and finiteness, but returns a JAX array because persisted values alone
cannot prove a metric law or recover erased provenance. Floating precision
follows the supplied values and caller configuration.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001

__all__ = [
    "NegativeEnergyFunctionalError",
    "NonfiniteEnergyFunctionalError",
    "sqrt_energy_functional",
]

#: Relative slack on the "materially negative" check: a functional assembled
#: from O(n²) float sums is allowed this much numerical undershoot below zero
#: before it counts as a sign error rather than rounding.
_NEGATIVE_TOLERANCE = 1e-12


class NonfiniteEnergyFunctionalError(ValueError):
    """Raised when the stored energy functional contains non-finite values."""

    def __init__(self) -> None:
        """Initialize the fixed non-finite-value diagnostic."""
        super().__init__("energy functional contains non-finite values")


class NegativeEnergyFunctionalError(ValueError):
    """Raised when the stored energy functional is materially negative."""

    def __init__(self) -> None:
        """Initialize the fixed invalid-sign diagnostic."""
        super().__init__("energy functional contains materially negative values")


@static_jaxtyping("*shape")
def sqrt_energy_functional(
    functional: ArrayLike,
) -> Float[Array, "*shape"]:
    """Return ``sqrt(S)`` after validating the stored functional ``S``.

    This artifact boundary validates values before taking the root and
    intentionally does not attach a mathematical law or construction origin.
    Use :func:`statadic.core.matrices.metrize_energy` while provenance and
    mathematical evidence are still available.

    Args:
        functional: Stored energy functional ``S``, with any shape.

    Returns:
        ``sqrt(S)`` as a JAX array, with entries inside the negative tolerance
        clamped to zero.

    Raises:
        NonfiniteEnergyFunctionalError: If any value is NaN or infinite.
        NegativeEnergyFunctionalError: If any value is negative by more than
            the relative tolerance — a sign error, not rounding.

    """
    values: Float[Array, "*shape"] = jnp.asarray(functional)
    if not bool(jnp.all(jnp.isfinite(values))):
        raise NonfiniteEnergyFunctionalError
    tolerance = _NEGATIVE_TOLERANCE * max(float(jnp.max(jnp.abs(values))), 1.0)
    if float(jnp.min(values)) < -tolerance:
        raise NegativeEnergyFunctionalError
    clamped = jnp.maximum(values, 0.0)
    return jnp.sqrt(clamped)
