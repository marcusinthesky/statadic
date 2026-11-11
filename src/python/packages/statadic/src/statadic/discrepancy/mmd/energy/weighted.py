# ruff: noqa: TC001  # Public runtime annotations are resolved by beartype.
"""Weighted rooted energy from direct mass-weighted mean ground distances.

Each block reduces ``sum_ik a_i b_k d(x_i, y_k)^alpha`` over positive-mass atoms,
as the unweighted packed core does with uniform masses. Routing through the
distance-induced kernel would anchor every Gram entry at one shared atom; far
from that anchor, float32 cancellation then corrupts nearby measures.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.core.axioms import Law
from statadic.core.domains import EmpiricalDistribution
from statadic.core.matrices import (  # evaluated runtime annotations
    DMat,
    HilbertianPseudometricLaw,
    MatrixProvenance,
    RootEnergyMetric,
    VStatisticScheme,
    _dmat,
)
from statadic.core.typing import Array  # runtime contract
from statadic.discrepancy.mmd.energy.geometry import (
    resolve_energy_geometry,
)
from statadic.discrepancy.mmd.weighted import _active_coordinates
from statadic.geometry.ground.config import GroundDistanceSelection
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    GroundDistance,
    cdist,
    squared_euclidean_distance,
)
from statadic.sample.packed import PackedClouds  # runtime contract

__all__ = [
    "weighted_energy_cross",
    "weighted_energy_matrix",
    "weighted_energy_squared_cross",
]


def _weighted_mean_distances(
    x: Array,
    mass_x: Array,
    y: Array,
    mass_y: Array,
    metric: GroundDistance[Law],
    power: float,
) -> tuple[Array, Array, Array]:
    """Return within-left, within-right and cross mass-weighted mean distances."""

    def expectation(a: Array, wa: Array, b: Array, wb: Array) -> Array:
        active = (wa[:, None] > 0) & (wb[None, :] > 0)
        if metric is EUCLIDEAN:
            # Select a finite argument before the power. Structural self-pairs
            # and coincident atoms use the zero subgradient convention.
            sq = squared_euclidean_distance(a, b)
            positive = active & (sq > 0)
            values = jnp.where(positive, jnp.where(positive, sq, 1) ** (power / 2), 0)
        else:
            values = jnp.where(active, cdist(a, b, metric=metric), 0) ** power
        return jnp.sum(values * wa[:, None] * wb[None, :])

    def self_expectation(a: Array, w: Array) -> Array:
        return expectation(a, w, a, w)

    def cross_row(a: Array, wa: Array) -> Array:
        def cross_entry(b: Array, wb: Array) -> Array:
            return expectation(a, wa, b, wb)

        return jax.vmap(cross_entry)(y, mass_y)

    within_x = jax.vmap(self_expectation)(x, mass_x)
    within_y = jax.vmap(self_expectation)(y, mass_y)
    cross = jax.vmap(cross_row)(x, mass_x)
    return within_x, within_y, cross


def weighted_energy_squared_cross(
    left: PackedClouds,
    right: PackedClouds,
    *,
    ground: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
) -> Array:
    """Return the unrooted weighted energy V-statistic for every pair.

    Expectations are V-statistics over atom masses; zero-mass atoms and
    inactive padding never enter a ground evaluation's reduction. A squared
    value below zero beyond roundoff on its own scale is unavailable (NaN).
    """
    if left.data.shape[-1] != right.data.shape[-1]:
        message = "Measure feature widths must agree."
        raise ValueError(message)
    geometry = resolve_energy_geometry(ground, exponent)
    metric, power = geometry.ground, geometry.exponent.value
    # Evaluate the coupled self and cross terms at one promoted input dtype.
    dtype = jnp.result_type(left.data, right.data, left.masses, right.masses)
    within_x, within_y, cross = _weighted_mean_distances(
        _active_coordinates(left).astype(dtype),
        left.masses.astype(dtype),
        _active_coordinates(right).astype(dtype),
        right.masses.astype(dtype),
        metric,
        power,
    )
    squared = 2.0 * cross - within_x[:, None] - within_y[None, :]
    scale = 2 * jnp.abs(cross) + jnp.abs(within_x[:, None]) + jnp.abs(within_y[None, :])
    tolerance = 32 * jnp.finfo(squared.dtype).eps * scale
    valid = (
        left.valid[:, None]
        & right.valid[None, :]
        & jnp.isfinite(squared)
        & (squared >= -tolerance)
    )
    return jnp.where(valid, jnp.maximum(squared, 0), jnp.nan)


def weighted_energy_cross(
    left: PackedClouds,
    right: PackedClouds,
    *,
    ground: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
) -> Array:
    """Return the rooted weighted energy distance."""
    return jnp.sqrt(
        weighted_energy_squared_cross(left, right, ground=ground, exponent=exponent)
    )


def weighted_energy_matrix(
    measures: PackedClouds,
    *,
    ground: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
) -> DMat[
    EmpiricalDistribution,
    MatrixProvenance[HilbertianPseudometricLaw, RootEnergyMetric[VStatisticScheme]],
]:
    """Produce a rooted weighted energy matrix at its actual V-statistic origin."""
    values = weighted_energy_cross(measures, measures, ground=ground, exponent=exponent)
    values = 0.5 * values + 0.5 * values.T
    diagonal = jnp.diag(values)
    values = values.at[jnp.diag_indices(measures.n_observations)].set(
        jnp.where(jnp.isfinite(diagonal), jnp.zeros_like(diagonal), diagonal)
    )
    return _dmat(values)
