# ruff: noqa: TC001  # Public runtime annotations are resolved by beartype.
"""Weighted V-statistic MMD using the existing declared point-kernel strategy."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.core.domains import EmpiricalDistribution
from statadic.core.matrices import (
    DMat,
    HilbertianPseudometricLaw,
    MatrixProvenance,
    VStatisticScheme,
    _dmat,
)
from statadic.core.typing import Array  # runtime numerical contract
from statadic.discrepancy.provenance import RootedMmd
from statadic.geometry.ground.similarities import (
    LINEAR_KERNEL,
    PositiveSemidefiniteKernelLaw,
    RBFKernel,
    SimilarityStrategy,
    gram,
)
from statadic.sample.packed import PackedClouds  # runtime contract

__all__ = [
    "measure_reference",
    "weighted_mmd_cross",
    "weighted_mmd_matrix",
    "weighted_mmd_squared_cross",
]


def _active_coordinates(packed: PackedClouds) -> Array:
    """Make arbitrary inactive padding inert before a point kernel sees it."""
    return packed.coordinates + 0.0


def measure_reference(measures: PackedClouds) -> Array:
    """Select one positive-mass nonzero anchor, shared by all Gram blocks.

    A nonzero anchor also supports grounds defined only away from the origin.
    If all positive-mass atoms are zero, the origin remains valid for Euclidean
    kernels; nonzero-domain kernels correctly return unavailable values.
    """
    coordinates = _active_coordinates(measures)
    eligible = (
        measures.valid[..., None]
        & (measures.masses > 0)
        & jnp.all(jnp.isfinite(coordinates), axis=-1)
        & jnp.any(coordinates != 0, axis=-1)
    )
    index = jnp.argmax(eligible.reshape(-1))
    return coordinates.reshape(-1, coordinates.shape[-1])[index]


def _linear_mmd_cross(
    x: Array, mass_x: Array, y: Array, mass_y: Array, reference: Array
) -> Array:
    """Use the explicit linear mean space instead of cancelling large Gram means."""
    mean_x = jnp.sum(
        jnp.where(mass_x[..., None] > 0, x - reference, 0) * mass_x[..., None], axis=1
    )
    mean_y = jnp.sum(
        jnp.where(mass_y[..., None] > 0, y - reference, 0) * mass_y[..., None], axis=1
    )
    squared = jnp.sum(jnp.square(mean_x[:, None] - mean_y[None, :]), axis=-1)
    return jnp.where(jnp.isfinite(squared), squared, jnp.nan)


def weighted_mmd_squared_cross(
    left: PackedClouds,
    right: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Array | None = None,
) -> Array:
    """Return weighted V-MMD squared, shape (left measures, right measures).

    Masses are probabilities established by the measure factory. All weighted
    reductions use ``where`` before multiplication, since a masked angular
    kernel may be undefined at an inactive zero vector. No U-statistic or
    silently debiased weighted estimate is constructed here.
    """
    if left.data.shape[-1] != right.data.shape[-1]:
        message = "Measure feature widths must agree."
        raise ValueError(message)
    if reference is None:
        reference = measure_reference(left)
    reference = jnp.asarray(reference)
    # Self and cross terms cancel, so every block must use the same ordinary
    # JAX promotion of its inputs; the retained measure arrays are unchanged.
    dtype = jnp.result_type(
        left.data,
        right.data,
        left.masses,
        right.masses,
        reference,
        *((kernel.bandwidth,) if isinstance(kernel, RBFKernel) else ()),
    )
    x = _active_coordinates(left).astype(dtype)
    y = _active_coordinates(right).astype(dtype)
    mass_x, mass_y = left.masses.astype(dtype), right.masses.astype(dtype)
    reference = reference.astype(dtype)
    measures_valid = left.valid[:, None] & right.valid[None, :]
    if kernel is LINEAR_KERNEL:
        return jnp.where(
            measures_valid,
            _linear_mmd_cross(x, mass_x, y, mass_y, reference),
            jnp.nan,
        )

    def expectation(a: Array, wa: Array, b: Array, wb: Array) -> Array:
        values = gram(a, b, kernel, reference=reference)
        active = (wa[:, None] > 0) & (wb[None, :] > 0)
        return jnp.sum(jnp.where(active, values, 0) * wa[:, None] * wb[None, :])

    def self_expectation(a: Array, w: Array) -> Array:
        return expectation(a, w, a, w)

    def cross_row(a: Array, wa: Array) -> Array:
        def cross_entry(b: Array, wb: Array) -> Array:
            return expectation(a, wa, b, wb)

        return jax.vmap(cross_entry)(y, mass_y)

    within_x = jax.vmap(self_expectation)(x, mass_x)
    within_y = jax.vmap(self_expectation)(y, mass_y)
    cross = jax.vmap(cross_row)(x, mass_x)
    squared = within_x[:, None] + within_y[None, :] - 2.0 * cross
    # Only cancellation on the arithmetic's own scale may be corrected. A
    # material violation of the declared PSD law remains an invalid lane.
    scale = jnp.abs(within_x[:, None]) + jnp.abs(within_y[None, :]) + 2 * jnp.abs(cross)
    tolerance = 32 * jnp.finfo(squared.dtype).eps * scale
    valid = measures_valid & jnp.isfinite(squared) & (squared >= -tolerance)
    return jnp.where(valid, squared, jnp.nan)


def weighted_mmd_cross(
    left: PackedClouds,
    right: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Array | None = None,
) -> Array:
    """Return rooted V-MMD; use the squared form for smooth zero-distance losses."""
    return jnp.sqrt(
        jnp.maximum(
            weighted_mmd_squared_cross(left, right, kernel=kernel, reference=reference),
            0,
        )
    )


def weighted_mmd_matrix(
    measures: PackedClouds,
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
) -> DMat[
    EmpiricalDistribution,
    MatrixProvenance[HilbertianPseudometricLaw, RootedMmd[VStatisticScheme]],
]:
    """Construct a law-bearing rooted V-MMD matrix from weighted measures.

    A declared PSD point kernel embeds probability measures into its Hilbert
    mean space. Separation is deliberately not claimed for arbitrary kernels.
    """
    values = weighted_mmd_cross(measures, measures, kernel=kernel)
    values = 0.5 * values + 0.5 * values.T
    diagonal = jnp.diag(values)
    values = values.at[jnp.diag_indices(measures.n_observations)].set(
        jnp.where(jnp.isfinite(diagonal), jnp.zeros_like(diagonal), diagonal)
    )
    return _dmat(values)
