# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Centered geometric operators, principal coordinates and frozen cross maps.

Classical scaling constructs ``-½ H D² H``. Euclidean embeddability requires
this observed Gram to be positive semidefinite; the squared-distance
negative-type law is a separate producer warrant, never inferred from a single
spectrum. Signed eigenvalues remain available even when that condition fails.

``pcoa_kernel`` and ``gram_coordinates`` retain fixed-width arrays under JIT and
batching. Positive coordinate columns and the Euclidean-validity diagnostic are
separate evidence. The host :class:`statadic.api.ordination.Ordination` applies its
explicit strict or positive-subspace policy and chooses presentation dimensions.
``CrossGramMap`` applies training centering to new rows without another solve.
Repeated eigenvalues, rank changes and selected-axis changes have no general
smooth coordinate derivative contract.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float  # noqa: TC002

from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001

__all__ = [
    "EIGENVALUE_TOLERANCE",
    "CrossGramMap",
    "NonEuclideanEmbeddingError",
    "OrdinationFit",
    "PCoAKernelResult",
    "center_gram",
    "gram_coordinates",
    "pcoa_kernel",
]

#: Relative slack on the non-negativity of the Gram spectrum. A *genuinely*
#: non-negative-type input produces a negative eigenvalue comparable in
#: magnitude to the leading positive one; anything smaller than this multiple of
#: ``max(|λ|, 1)`` is finite-sample estimation noise in an *empirical* distance
#: whose population counterpart is exactly of negative type. The same constant
#: sets the rank cut, so the axes that are kept are strictly positive.
EIGENVALUE_TOLERANCE: Final = 1e-8


class NonEuclideanEmbeddingError(ValueError):
    """Raised when the PCoA Gram spectrum is materially negative.

    A ``ValueError`` subclass so the pre-t46 ``raise ValueError`` contract at
    Paper 1's DISCO call site still holds for any caller catching the base
    class.
    """

    def __init__(self, min_eigenvalue: float, max_eigenvalue: float) -> None:
        """Record the offending spectrum bounds in the message.

        Args:
            min_eigenvalue: Smallest Gram eigenvalue observed.
            max_eigenvalue: Scale the tolerance was taken relative to —
                ``max(|λ_max|, 1)``.

        """
        super().__init__(
            "PCoA produced a negative eigenvalue beyond numerical tolerance "
            f"({min_eigenvalue:.3e}) relative to the leading eigenvalue "
            f"({max_eigenvalue:.3e}); expected approximately non-negative "
            "(negative-type distance)."
        )


class PCoAKernelResult(NamedTuple):
    """Fixed-structure result of :func:`pcoa_kernel`.

    Inactive coordinate columns are zero. Consumers inside a transform retain
    the full fixed-width carrier and use ``active``; eager presentation doors
    may select those columns after leaving the trace.
    """

    coordinates: Float[Array, "*batch n n"]
    eigenvalues: Float[Array, "*batch n"]
    active: Bool[Array, "*batch n"]
    minimum_eigenvalue: Float[Array, "*batch"]
    spectral_scale: Float[Array, "*batch"]
    valid: Bool[Array, "*batch"]


class OrdinationFit(NamedTuple):
    """Retained selected coordinates and full signed operator spectrum."""

    coordinates: Array
    eigenvalues: Array
    threshold: Array
    euclidean: Array


def center_gram(gram: Array) -> Array:
    """Double-center a Gram block without materializing a centering matrix."""
    return (
        gram
        - jnp.mean(gram, axis=-1, keepdims=True)
        - jnp.mean(gram, axis=-2, keepdims=True)
        + jnp.mean(gram, axis=(-2, -1), keepdims=True)
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossGramMap:
    """A frozen training-centered cross-Gram coordinate map.

    ``axes`` is the retained eigenvector matrix divided by square-root
    eigenvalues. All fields are numerical leaves; applying the map never
    learns from the query batch.
    """

    column_mean: Array
    grand_mean: Array
    axes: Array

    def transform(self, cross_gram: Array) -> Array:
        """Center each query against training statistics and apply fixed axes."""
        centered = (
            cross_gram
            - jnp.mean(cross_gram, axis=-1, keepdims=True)
            - self.column_mean
            + self.grand_mean
        )
        return centered @ self.axes


def gram_coordinates(
    gram: Array, tolerance: float = EIGENVALUE_TOLERANCE
) -> PCoAKernelResult:
    """Solve centered symmetric geometry with fixed-width signed evidence.

    Positive coordinate columns represent the positive spectral subspace.
    ``valid`` separately reports Euclidean embeddability at the requested
    tolerance; positive-subspace approximation never changes that diagnostic.
    """
    centered = center_gram(gram)
    centered = (centered + jnp.swapaxes(centered, -1, -2)) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(centered)
    eigenvalues = eigenvalues[..., ::-1]
    eigenvectors = eigenvectors[..., :, ::-1]
    spectral_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues), axis=-1), 1.0)
    minimum = jnp.min(eigenvalues, axis=-1)
    active = eigenvalues > tolerance * spectral_scale[..., None]
    coordinates = (
        eigenvectors * jnp.sqrt(jnp.where(active, eigenvalues, 0))[..., None, :]
    )
    return PCoAKernelResult(
        coordinates,
        eigenvalues,
        active,
        minimum,
        spectral_scale,
        jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        & (minimum >= -tolerance * spectral_scale),
    )


@static_jaxtyping("*batch n")
def pcoa_kernel(
    distances: Float[Array, "*batch n n"],
    tolerance: float = EIGENVALUE_TOLERANCE,
) -> PCoAKernelResult:
    """Return a fixed-width classical-scaling embedding on the JAX graph.

    The kernel never raises or changes output shape. ``valid`` records whether
    the Gram spectrum satisfies the negative-type precondition within the
    requested tolerance; eager doors turn a false diagnostic into
    :class:`NonEuclideanEmbeddingError`.
    """
    values = jnp.asarray(distances)
    return gram_coordinates(-0.5 * values**2, tolerance)
