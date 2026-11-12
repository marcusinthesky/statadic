# ruff: noqa: TC001  # Public runtime annotations are resolved by beartype.
"""Qualified pairwise transport for weighted empirical measures.

Only real-line W2 and finite shared-projection sliced W2 receive a Hilbertian
pseudometric producer. Other transport distances do not inherit these laws.
"""

from __future__ import annotations

import jax.numpy as jnp

from statadic.core.domains import EmpiricalDistribution, TransportOrigin
from statadic.core.matrices import (
    DMat,
    HilbertianPseudometricLaw,
    MatrixProvenance,
    _dmat,
)
from statadic.core.typing import Array  # runtime contract
from statadic.discrepancy.transport.sliced import (
    ProjectionDesign,
    sliced_wasserstein_under_design,
)
from statadic.discrepancy.transport.univariate import wasserstein_1d
from statadic.sample.packed import PackedClouds  # runtime contract

__all__ = [
    "FiniteSlicedW2Origin",
    "QuantileW2Origin",
    "finite_sliced_w2_matrix",
    "quantile_w2_matrix",
    "transport_cross",
]

_HILBERT_ORDERS = (2.0,)


class QuantileW2Origin(TransportOrigin):
    """Real-line W2, the L2 distance between probability quantile functions."""


class FiniteSlicedW2Origin(TransportOrigin):
    """Finite direct sum of quantile L2 spaces under one shared projection design."""


def transport_cross(
    left: PackedClouds,
    right: PackedClouds,
    *,
    design: ProjectionDesign | None = None,
) -> Array:
    """Evaluate weighted unequal-support transport, shape (left, right).

    Without a sliced design the support must be real-line and the order is two.
    A design explicitly owns its order and shared projections.
    """
    if left.data.shape[-1] != right.data.shape[-1]:
        message = "Measure feature widths must agree."
        raise ValueError(message)
    if design is None and left.data.shape[-1] != 1:
        message = "Wasserstein1D requires one coordinate per atom."
        raise ValueError(message)

    if design is not None and jnp.result_type(
        left.data.dtype, left.masses.dtype
    ) != jnp.result_type(right.data.dtype, right.masses.dtype):
        message = (
            "Shared sliced geometry requires matching effective coordinate dtypes; "
            "query precision cannot redraw the retained projection design."
        )
        raise ValueError(message)

    def pair(x: PackedClouds, y: PackedClouds) -> Array:
        if design is None:
            return wasserstein_1d(x, y, p=2)
        return sliced_wasserstein_under_design(design, x, y)

    return left.map_cross(right, pair)


def quantile_w2_matrix(
    measures: PackedClouds,
) -> DMat[
    EmpiricalDistribution, MatrixProvenance[HilbertianPseudometricLaw, QuantileW2Origin]
]:
    """Produce W2 distances via the exact quantile Hilbert embedding."""
    values = transport_cross(measures, measures)
    values = (
        (0.5 * values + 0.5 * values.T)
        .at[jnp.diag_indices(measures.n_observations)]
        .set(
            jnp.where(
                jnp.isfinite(jnp.diag(values)),
                jnp.zeros_like(jnp.diag(values)),
                jnp.diag(values),
            )
        )
    )
    return _dmat(values)


def finite_sliced_w2_matrix(
    measures: PackedClouds, design: ProjectionDesign
) -> DMat[
    EmpiricalDistribution,
    MatrixProvenance[HilbertianPseudometricLaw, FiniteSlicedW2Origin],
]:
    """Produce finite sliced W2 with a conservative nonseparating Hilbert law."""
    if float(design.p) not in _HILBERT_ORDERS:
        message = "Only order-two sliced transport has this Hilbertian producer."
        raise ValueError(message)
    values = transport_cross(measures, measures, design=design)
    values = (
        (0.5 * values + 0.5 * values.T)
        .at[jnp.diag_indices(measures.n_observations)]
        .set(
            jnp.where(
                jnp.isfinite(jnp.diag(values)),
                jnp.zeros_like(jnp.diag(values)),
                jnp.diag(values),
            )
        )
    )
    return _dmat(values)
