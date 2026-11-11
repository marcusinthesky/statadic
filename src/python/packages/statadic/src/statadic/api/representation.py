# ruff: noqa: TC001  # Public runtime annotations are resolved by beartype.
"""Explicit empirical-measure geometry shared by analysis and prediction.

Prepared geometry freezes its reference observations and its concrete numerical
specification. Cross evaluation is always new observations by reference rows.
Pairwise law evidence comes from qualified scientific producers, never from
an observed spectrum or a caller-supplied Boolean.
"""

from __future__ import annotations

__all__ = [
    "MMD",
    "Energy",
    "GeometrySpec",
    "HilbertianMeasureMatrix",
    "NativeMeasureMatrix",
    "ObservedDissimilarity",
    "PairwiseGeometry",
    "PreparedGeometry",
    "RowId",
    "SinkhornDivergence",
    "SlicedWasserstein",
    "Wasserstein1D",
]

import math
from collections.abc import Sequence  # runtime public contract  # noqa: TC003
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from statadic.api.data import MeasureColumn
from statadic.core.domains import ConstructionOrigin, EmpiricalDistribution
from statadic.core.matrices import DMat, HilbertianPseudometricLaw, MatrixProvenance
from statadic.core.typing import Array, ArrayLike  # runtime public contract
from statadic.discrepancy.mmd.energy.geometry import resolve_energy_geometry
from statadic.discrepancy.mmd.energy.weighted import (
    weighted_energy_cross,
    weighted_energy_matrix,
    weighted_energy_squared_cross,
)
from statadic.discrepancy.mmd.weighted import (
    measure_reference,
    weighted_mmd_cross,
    weighted_mmd_matrix,
    weighted_mmd_squared_cross,
)
from statadic.discrepancy.transport.matrix import (
    finite_sliced_w2_matrix,
    quantile_w2_matrix,
    transport_cross,
)
from statadic.discrepancy.transport.sinkhorn import sinkhorn_cost
from statadic.discrepancy.transport.sliced import (
    ProjectionDesign,
    sliced_wasserstein_power_under_design,
)
from statadic.discrepancy.transport.univariate import wasserstein_1d_power
from statadic.geometry.ground.config import (
    GroundDistanceSelection,  # runtime annotations
)
from statadic.geometry.ground.similarities import (
    PositiveSemidefiniteKernelLaw,
    SimilarityStrategy,
)
from statadic.sample.packed import PackedClouds, checked_packed_clouds

type NativeMeasureMatrix = DMat[
    EmpiricalDistribution,
    MatrixProvenance[HilbertianPseudometricLaw, ConstructionOrigin],
]
type RowId = int | str
_HILBERT_ORDERS = (2.0,)
_MATRIX_NDIM = 2
_MIN_ROWS = 2


@dataclass(frozen=True)
class ObservedDissimilarity:
    """A checked observed matrix and ordered identities, without population laws."""

    values: Array
    row_ids: tuple[RowId, ...]
    spec: str = "observed matrix only"

    @classmethod
    def from_array(
        cls, dissimilarities: ArrayLike, row_ids: Sequence[RowId] | None = None
    ) -> ObservedDissimilarity:
        """Own finite, nonnegative symmetric zero-diagonal values and unique IDs."""
        values = jnp.array(dissimilarities, copy=True)
        if (
            values.ndim != _MATRIX_NDIM
            or values.shape[0] != values.shape[1]
            or len(values) < _MIN_ROWS
        ):
            message = "Dissimilarities require a square matrix with at least two rows."
            raise ValueError(message)
        if not jnp.issubdtype(values.dtype, jnp.number) or jnp.iscomplexobj(values):
            message = "Dissimilarities must be real numeric values."
            raise TypeError(message)
        values = values.astype(jnp.result_type(values, 1.0))
        slack = 32 * jnp.finfo(values.dtype).eps * jnp.maximum(jnp.max(values), 1)
        if not bool(
            jnp.all(jnp.isfinite(values))
            & jnp.all(values >= 0)
            & jnp.all(jnp.abs(values - values.T) <= slack)
            & jnp.all(jnp.abs(jnp.diag(values)) <= slack)
        ):
            message = (
                "Dissimilarities must be finite, nonnegative, symmetric "
                "and zero-diagonal."
            )
            raise ValueError(message)
        ids = tuple(range(len(values))) if row_ids is None else tuple(row_ids)
        if (
            len(ids) != len(values)
            or len(set(ids)) != len(ids)
            or any(
                isinstance(key, bool) or not isinstance(key, (int, str)) for key in ids
            )
        ):
            message = "Dissimilarity IDs must be unique aligned integers or strings."
            raise ValueError(message)
        return cls(values, ids)


@dataclass(frozen=True)
class HilbertianMeasureMatrix:
    """A producer's explicit assertion that a measure matrix is Hilbertian.

    ``construction`` describes the producer's warrant; it is a declaration,
    not a proof inferred from the observed matrix or its phantom annotations.
    """

    matrix: NativeMeasureMatrix
    construction: str

    def __post_init__(self) -> None:
        """Require an actual native matrix and a named construction warrant."""
        if not isinstance(self.matrix, DMat):
            message = "A Hilbertian measure matrix requires a native DMat."
            raise TypeError(message)
        if not isinstance(self.construction, str) or not self.construction.strip():
            message = "A Hilbertian measure matrix requires a construction warrant."
            raise ValueError(message)

    @property
    def values(self) -> Array:
        """Return the native matrix values."""
        return self.matrix.values


@runtime_checkable
class GeometrySpec(Protocol):
    """Concrete geometry evaluates weighted measures and owns its law producer."""

    def prepare(
        self, measures: PackedClouds, *, row_ids: Sequence[RowId] | None = None
    ) -> PreparedGeometry:
        """Freeze reference measures and their ordered identities."""
        ...

    def cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate a left-by-right distance matrix."""
        ...

    def matrix(self, measures: PackedClouds) -> HilbertianMeasureMatrix | None:
        """Declare a construction warrant when this method supports it."""
        ...


def _prepare(
    spec: GeometrySpec,
    measures: PackedClouds | MeasureColumn,
    row_ids: Sequence[RowId] | None,
) -> PreparedGeometry:
    if isinstance(measures, MeasureColumn):
        ids = measures.row_ids if row_ids is None else tuple(row_ids)
        if ids != measures.row_ids:
            message = "Explicit row_ids must equal the MeasureColumn identities."
            raise ValueError(message)
        measures = measures.measures
    else:
        ids = (
            tuple(range(measures.n_observations)) if row_ids is None else tuple(row_ids)
        )
    return PreparedGeometry(spec, measures, ids)


@dataclass(frozen=True)
class MMD:
    """Rooted probability-weighted V-MMD under a declared PSD point kernel."""

    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw]
    estimator: str = field(default="v", init=False)
    rooted: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        """Reject undeclared kernel callables."""
        if not isinstance(self.kernel, SimilarityStrategy):
            message = "MMD requires a declared PSD SimilarityStrategy."
            raise TypeError(message)

    def squared_cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate squared measure distances without a rooted self-distance."""
        return weighted_mmd_squared_cross(
            left, right, kernel=self.kernel, reference=measure_reference(right)
        )

    def prepare(
        self,
        measures: PackedClouds | MeasureColumn,
        *,
        row_ids: Sequence[RowId] | None = None,
    ) -> PreparedGeometry:
        """Freeze the kernel and reference measure order."""
        return _prepare(self, measures, row_ids)

    def cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate weighted rooted V-MMD for every left/right pair."""
        return weighted_mmd_cross(
            left, right, kernel=self.kernel, reference=measure_reference(right)
        )

    def matrix(self, measures: PackedClouds) -> HilbertianMeasureMatrix:
        """Use the scientific producer's Hilbertian mean-embedding guarantee."""
        return HilbertianMeasureMatrix(
            weighted_mmd_matrix(measures, kernel=self.kernel),
            "Rooted V-MMD from a declared PSD kernel mean embedding.",
        )


@dataclass(frozen=True)
class Energy:
    """Rooted V-energy with an explicitly qualified ground and exponent."""

    ground: GroundDistanceSelection = "euclidean"
    exponent: int | float = 1.0
    estimator: str = field(default="v", init=False)
    rooted: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        """Close the actual ground/exponent compatibility before evaluation."""
        resolve_energy_geometry(self.ground, float(self.exponent))

    def squared_cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate squared measure distances without a rooted self-distance."""
        return weighted_energy_squared_cross(
            left, right, ground=self.ground, exponent=float(self.exponent)
        )

    def prepare(
        self,
        measures: PackedClouds | MeasureColumn,
        *,
        row_ids: Sequence[RowId] | None = None,
    ) -> PreparedGeometry:
        """Freeze energy geometry and the reference measure order."""
        return _prepare(self, measures, row_ids)

    def cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate rooted energy from direct mass-weighted ground distances."""
        return weighted_energy_cross(
            left, right, ground=self.ground, exponent=float(self.exponent)
        )

    def matrix(self, measures: PackedClouds) -> HilbertianMeasureMatrix:
        """Preserve the energy V-statistic origin in the qualified matrix."""
        return HilbertianMeasureMatrix(
            weighted_energy_matrix(
                measures, ground=self.ground, exponent=float(self.exponent)
            ),
            "Rooted V-energy from a qualified negative-type ground geometry.",
        )


@dataclass(frozen=True)
class Wasserstein1D:
    """Exact weighted real-line W2, a quantile Hilbert-space distance."""

    p: int | float

    def __post_init__(self) -> None:
        """Keep the initially qualified Hilbertian family explicit."""
        if self.p not in _HILBERT_ORDERS:
            message = "Wasserstein1D currently qualifies p=2 only."
            raise ValueError(message)

    @staticmethod
    def squared_cross(left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate squared measure distances without a rooted self-distance."""
        return left.map_cross(right, lambda x, y: wasserstein_1d_power(x, y, p=2))

    def prepare(
        self,
        measures: PackedClouds | MeasureColumn,
        *,
        row_ids: Sequence[RowId] | None = None,
    ) -> PreparedGeometry:
        """Validate the real-line domain and freeze reference observations."""
        prepared = _prepare(self, measures, row_ids)
        if prepared.reference.data.shape[-1] != 1:
            message = "Wasserstein1D requires one coordinate per atom."
            raise ValueError(message)
        return prepared

    @staticmethod
    def cross(left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate exact weighted W2 for unequal atom counts."""
        return transport_cross(left, right)

    @staticmethod
    def matrix(measures: PackedClouds) -> HilbertianMeasureMatrix:
        """Retain the quantile Hilbertian producer's actual origin."""
        return HilbertianMeasureMatrix(
            quantile_w2_matrix(measures), "Real-line W2 quantile Hilbert embedding."
        )


@dataclass(frozen=True)
class SlicedWasserstein:
    """Finite transport pseudometric under an explicit shared projection design."""

    design: ProjectionDesign

    def squared_cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate squared measure distances without a rooted self-distance."""
        power = left.map_cross(
            right, lambda x, y: sliced_wasserstein_power_under_design(self.design, x, y)
        )
        # The squared distance has zero coordinate derivative at identical
        # measures. Keep a fractional outer power away from its singularity.
        zero = power == 0
        return jnp.where(zero, 0, jnp.where(zero, 1, power) ** (2 / self.design.p))

    def prepare(
        self,
        measures: PackedClouds | MeasureColumn,
        *,
        row_ids: Sequence[RowId] | None = None,
    ) -> PreparedGeometry:
        """Freeze the single projection design used by every comparison."""
        return _prepare(self, measures, row_ids)

    def cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate the common finite design for every left/right pair."""
        return transport_cross(left, right, design=self.design)

    def matrix(self, measures: PackedClouds) -> HilbertianMeasureMatrix | None:
        """Only p=2 receives the finite direct-sum Hilbertian guarantee."""
        if float(self.design.p) not in _HILBERT_ORDERS:
            return None
        return HilbertianMeasureMatrix(
            finite_sliced_w2_matrix(measures, self.design),
            "Finite shared W2 projections form a direct-sum Hilbert embedding.",
        )


@dataclass(frozen=True)
class SinkhornDivergence:
    """Root debiased entropic OT on the squared-Euclidean ground-cost scale."""

    epsilon: float
    iterations: int = 128
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        """Validate regularization, marginal tolerance and iteration budget."""
        valid_scalars = all(
            math.isfinite(x) and x > 0 for x in (self.epsilon, self.tolerance)
        )
        if not valid_scalars or type(self.iterations) is not int or self.iterations < 1:
            msg = "Sinkhorn epsilon, tolerance and iteration count must be positive."
            raise ValueError(msg)

    def prepare(
        self,
        measures: PackedClouds | MeasureColumn,
        *,
        row_ids: Sequence[RowId] | None = None,
    ) -> PreparedGeometry:
        """Freeze solver settings and ordered reference measures."""
        return _prepare(self, measures, row_ids)

    def squared_cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Subtract both self-costs before taking any root."""

        def solve(a: PackedClouds, b: PackedClouds) -> Array:
            return sinkhorn_cost(
                a,
                b,
                epsilon=self.epsilon,
                iterations=self.iterations,
                tolerance=self.tolerance,
            )

        within_left = jax.vmap(solve)(left, left)
        within_right = jax.vmap(solve)(right, right)
        cross = left.map_cross(right, solve)
        bias = (within_left[:, None] + within_right[None, :]) / 2
        value = cross - bias
        slack = self.tolerance * (1 + jnp.abs(cross) + jnp.abs(bias))
        return jnp.where(value >= -slack, jnp.maximum(value, 0), jnp.nan)

    def cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate the root divergence, including unavailable solver results."""
        return jnp.sqrt(self.squared_cross(left, right))

    @staticmethod
    def matrix(measures: PackedClouds) -> None:  # noqa: ARG004 -- geometry protocol
        """No Hilbertian warrant is declared for this numerical geometry."""
        return


@dataclass(frozen=True)
class PairwiseGeometry:
    """A produced matrix, ordered row identities and its retained geometry.

    Law flags are derived from the native scientific producer. Direct public
    construction cannot attach unsupported mathematical claims to an array.
    """

    values: Array
    row_ids: tuple[RowId, ...]
    spec: GeometrySpec
    matrix: NativeMeasureMatrix | None
    _witness: HilbertianMeasureMatrix | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Require the producer boundary rather than arbitrary evidence flags."""
        message = "PairwiseGeometry must be produced by PreparedGeometry.pairwise()."
        raise TypeError(message)

    @classmethod
    def _from_prepared(
        cls,
        values: Array,
        row_ids: tuple[RowId, ...],
        spec: GeometrySpec,
        witness: HilbertianMeasureMatrix | None,
    ) -> PairwiseGeometry:
        """Create a coherent carrier at the checked producer boundary."""
        result = object.__new__(cls)
        object.__setattr__(
            result, "values", values if witness is None else witness.values
        )
        object.__setattr__(result, "row_ids", row_ids)
        object.__setattr__(result, "spec", spec)
        object.__setattr__(
            result, "matrix", None if witness is None else witness.matrix
        )
        object.__setattr__(result, "_witness", witness)
        return result

    @property
    def construction_warrant(self) -> str | None:
        """Return the producer's explicit construction declaration, if present."""
        return None if self._witness is None else self._witness.construction

    @property
    def negative_type(self) -> bool:
        """Whether the concrete producer supplies a negative-type law."""
        return self._witness is not None

    @property
    def squared_negative_type(self) -> bool:
        """Whether squared distances have the producer's Hilbertian guarantee."""
        return self._witness is not None


@dataclass(frozen=True)
class PreparedGeometry:
    """Frozen reference measures and numerical specification for future queries."""

    spec: GeometrySpec
    reference: PackedClouds
    row_ids: tuple[RowId, ...]

    def __post_init__(self) -> None:
        """Require the structural geometry protocol and aligned identities."""
        if self.reference.counts.ndim != 1:
            message = "Prepared geometry requires one collection axis."
            raise ValueError(message)
        checked_packed_clouds(
            self.reference.data,
            self.reference.counts,
            self.reference.masses,
            require_positive_counts=True,
        )
        if not isinstance(self.spec, GeometrySpec):
            message = "PreparedGeometry requires a GeometrySpec implementation."
            raise TypeError(message)
        if any(
            isinstance(key, bool) or not isinstance(key, (int, str))
            for key in self.row_ids
        ):
            message = "Prepared reference row_ids must be integers or strings."
            raise ValueError(message)
        if len(self.row_ids) != self.reference.n_observations or len(
            set(self.row_ids)
        ) != len(self.row_ids):
            message = "Prepared reference row_ids must be unique and aligned."
            raise ValueError(message)

    def cross(self, other: PackedClouds) -> Array:
        """Return distances of shape (new observations, reference observations)."""
        values = jnp.asarray(self.spec.cross(other, self.reference))
        expected = (other.n_observations, self.reference.n_observations)
        if values.shape != expected:
            message = (
                f"Geometry cross() must return shape {expected} "
                f"(new, reference observations); got {values.shape}."
            )
            raise ValueError(message)
        if not bool(jnp.all(jnp.isfinite(values))):
            message = "Prepared geometry is undefined on these measure supports."
            raise ValueError(message)
        return values

    def pairwise(self) -> PairwiseGeometry:
        """Evaluate the retained reference set without inferring laws from values."""
        witness = self.spec.matrix(self.reference)
        if witness is not None and not isinstance(witness, HilbertianMeasureMatrix):
            message = "Geometry matrix requires a HilbertianMeasureMatrix warrant."
            raise TypeError(message)
        values = witness.values if witness is not None else self.cross(self.reference)
        if not bool(jnp.all(jnp.isfinite(values))):
            message = "Prepared geometry is undefined on these measure supports."
            raise ValueError(message)
        return PairwiseGeometry._from_prepared(  # noqa: SLF001
            values, self.row_ids, self.spec, witness
        )
