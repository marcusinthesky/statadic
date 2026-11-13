# jaxtyping numerical carrier contracts.
"""MMD kernel constructions whose validity depends on discrepancy theory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import jax

from statadic.core.axioms import Law
from statadic.discrepancy.mmd.energy.geometry import (
    EnergyExponent,
    SeparatingDistributionEnergyLaw,
)
from statadic.geometry.ground.similarities import (
    CharacteristicKernelProperties,
    PositiveSemidefiniteKernelLaw,
    PositiveSemidefiniteKernelProperties,
    distance_induced_gram,
)

if TYPE_CHECKING:
    from statadic.core.domains import (
        ClaimAuthority,
        SpaceDescriptor,
        ValueEvidence,
    )
    from statadic.core.transforms import TransformDescriptor
    from statadic.discrepancy.mmd.energy.geometry import (
        EnergyGeometry,
    )
    from statadic.geometry.ground.similarities import SimilarityStrategy

__all__ = ["distance_induced_kernel"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _DistanceInducedKernel[
    KernelLawT: PositiveSemidefiniteKernelLaw,
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
]:
    """Validated negative-type ground transformed into a PSD point kernel."""

    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT]
    _law: KernelLawT = field(metadata={"static": True})

    @property
    def name(self) -> str:
        return (
            f"distance_induced[{self.geometry.ground.name},"
            f"alpha={self.geometry.exponent.value}]"
        )

    @property
    def law(self) -> KernelLawT:
        return self._law

    @property
    def space(self) -> SpaceDescriptor:
        return self.geometry.ground.space

    @property
    def transform(self) -> TransformDescriptor:
        return self.geometry.ground.transform

    @property
    def authority(self) -> ClaimAuthority:
        return self.geometry.ground.authority

    @property
    def required_evidence(self) -> type[ValueEvidence]:
        return self.geometry.ground.required_evidence

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        *,
        reference: jax.Array | None = None,
    ) -> jax.Array:
        return distance_induced_gram(
            x,
            y,
            metric=self.geometry.ground,
            exponent=self.geometry.exponent.value,
            reference=reference,
        )


@overload
def distance_induced_kernel[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
](
    geometry: EnergyGeometry[
        GroundLawT,
        ExponentT,
        SeparatingDistributionEnergyLaw,
    ],
) -> SimilarityStrategy[CharacteristicKernelProperties]: ...


@overload
def distance_induced_kernel[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT],
) -> SimilarityStrategy[PositiveSemidefiniteKernelProperties]: ...


def distance_induced_kernel[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT],
) -> SimilarityStrategy[PositiveSemidefiniteKernelLaw]:
    """Construct the unhalved energy kernel from one compatible geometry.

    The factory has no independent ground or exponent arguments. A separating
    energy law yields a characteristic kernel; an ordinary negative-type law
    yields only PSD. Unsupported ground-power pairs cannot be represented by an
    :class:`EnergyGeometry`, so they fail before this construction.
    """
    if isinstance(geometry.law, SeparatingDistributionEnergyLaw):
        return _DistanceInducedKernel(geometry, CharacteristicKernelProperties())
    return _DistanceInducedKernel(geometry, PositiveSemidefiniteKernelProperties())
