"""Typed mathematical transforms and static-preserving JAX doors.

The map is part of the theorem
------------------------------
A law is never transported by formula alone.  Its source carrier, target
carrier, equality, totality requirement, and construction origin determine
what survives composition.  In particular, L2 normalization is total only on
nonzero vectors and is not injective for coordinate equality.  Pulling a
sphere metric back through it therefore yields a pseudometric on raw nonzero
vectors, while negative type is preserved.

The public functions below make the permitted proof steps explicit.  There is
deliberately no generic ``pullback`` returning its input law: such a function
would allow a separating target metric to cross a noninjective map unchanged.
Only an :class:`InjectiveTransform` can use :func:`restrict_law`, and quotient
metrization additionally requires an explicit fibre/equality witness.

``typed_jit`` preserves a callable's signature, including its semantic return
type. ``typed_vmap`` and ``map_arrays`` instead describe the actual rank change:
one mapped input axis becomes one leading output axis. Their shape contract is
for a single array input and array output. Structured pytrees require their own
leaf-shape contract rather than a cast claiming an unchanged signature.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003  # evaluated runtime annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, final

import jax
from jaxtyping import Shaped  # noqa: TC002

from statadic.core.domains import (
    ConstructionOrigin,
    L2UnitSphereSpace,
    NonzeroVectorCarrierSpace,
    NormalizationPullbackOrigin,
    OriginT,
    SpaceDescriptor,
    SpaceLaw,
)
from statadic.core.typing import (  # noqa: TC001  # evaluated runtime annotations
    DimensionVar,
    Elements,
    Shape,
    ShapedArray,
)

__all__ = [
    "EmpiricalPushforwardMap",
    "IdentityMap",
    "InjectiveTransform",
    "L2Normalization",
    "L2NormalizationContract",
    "MapKind",
    "NonzeroInputRequired",
    "PartialDomainRequirement",
    "PopulationPushforwardMap",
    "PositiveRayQuotient",
    "ProjectiveQuotient",
    "QuotientMetricWitness",
    "QuotientTransform",
    "RestrictionMap",
    "TotalOnDeclaredSource",
    "TransformContract",
    "TransformDescriptor",
    "map_arrays",
    "pullback_negative_type",
    "pullback_pseudometric",
    "quotient_metric",
    "restrict_law",
    "typed_jit",
    "typed_vmap",
]

if TYPE_CHECKING:
    from statadic.core.axioms import (
        Law,
        MetricLaw,
        NegativeTypeLaw,
        PseudometricLaw,
    )


class MapKind:
    """Root marker for a mathematical map's identity."""


class IdentityMap(MapKind):
    """The identity map on one declared mathematical space."""


class RestrictionMap(MapKind):
    """Inclusion of a restricted carrier into a wider carrier."""


class L2Normalization(MapKind):
    """The map ``x ↦ x / ‖x‖₂`` on finite nonzero vectors."""


class PositiveRayQuotient(MapKind):
    """Quotient map identifying strictly positive scalar multiples."""


class ProjectiveQuotient(MapKind):
    """Quotient map identifying all nonzero scalar multiples."""


class EmpiricalPushforwardMap(MapKind):
    """Image of a finite empirical measure under a declared point map."""


class PopulationPushforwardMap(MapKind):
    """Image of a population probability law under a declared point map."""


class PartialDomainRequirement:
    """Root marker for the totality condition recorded by a transform."""


class TotalOnDeclaredSource(PartialDomainRequirement):
    """The map is total on every value of its declared source space."""


class NonzeroInputRequired(PartialDomainRequirement):
    """The map requires a finite input whose L2 norm is nonzero."""


SourceSpaceInT = TypeVar(  # noqa: PLC0105
    "SourceSpaceInT",
    bound=SpaceDescriptor,
    contravariant=True,
)
TargetSpaceT = TypeVar(  # noqa: PLC0105
    "TargetSpaceT",
    bound=SpaceDescriptor,
    covariant=True,
)
MapT = TypeVar("MapT", bound=MapKind, covariant=True)  # noqa: PLC0105
RequirementT = TypeVar(  # noqa: PLC0105
    "RequirementT",
    bound=PartialDomainRequirement,
    covariant=True,
)
QuotientSourceT = TypeVar(
    "QuotientSourceT",
    bound=SpaceDescriptor,
)
QuotientTargetT = TypeVar(
    "QuotientTargetT",
    bound=SpaceDescriptor,
)
QuotientMapT = TypeVar("QuotientMapT", bound=MapKind)
QuotientRequirementT = TypeVar(
    "QuotientRequirementT",
    bound=PartialDomainRequirement,
)
QuotientOriginT = TypeVar(
    "QuotientOriginT",
    bound=ConstructionOrigin,
)


@dataclass(frozen=True, slots=True)
class TransformDescriptor:
    """Non-generic root for a complete static transform descriptor."""


class TransformContract(
    TransformDescriptor,
    Generic[SourceSpaceInT, TargetSpaceT, MapT, RequirementT, OriginT],  # noqa: UP046
):
    """Zero-field descriptor of a source-to-target mathematical map.

    All five parameters are static facts.  In particular, the requirement is
    not inferred from an array and the origin is construction provenance, not
    claim authority.  Eager doors must establish any value evidence needed to
    enter the declared source carrier.
    """


@final
@dataclass(frozen=True, slots=True)
class InjectiveTransform(
    TransformContract[SourceSpaceInT, TargetSpaceT, MapT, RequirementT, OriginT],
    Generic[SourceSpaceInT, TargetSpaceT, MapT, RequirementT, OriginT],  # noqa: UP046
):
    """A transform injective for the source and target equalities."""


@final
@dataclass(frozen=True, slots=True)
class QuotientTransform(
    TransformContract[
        QuotientSourceT,
        QuotientTargetT,
        QuotientMapT,
        QuotientRequirementT,
        QuotientOriginT,
    ],
    Generic[  # noqa: UP046
        QuotientSourceT,
        QuotientTargetT,
        QuotientMapT,
        QuotientRequirementT,
        QuotientOriginT,
    ],
):
    """Invariant quotient descriptor, preventing cross-quotient witnesses."""


type L2NormalizationContract = TransformContract[
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    L2Normalization,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
]


@final
@dataclass(frozen=True, slots=True)
class QuotientMetricWitness(
    Generic[QuotientSourceT, QuotientTargetT, QuotientMapT]  # noqa: UP046
):
    """Witness that zero-distance fibres are exactly the target equality.

    This is intentionally constructible only as a declaration: Python types do
    not prove the theorem.  Its presence nevertheless prevents a quotient map
    from silently upgrading a pseudometric to a metric.
    """


def pullback_pseudometric[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, PseudometricLaw],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, PseudometricLaw]:
    """Pull a pseudometric back, deliberately making no separation claim."""
    del law, transform
    return SpaceLaw()


def pullback_negative_type[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, NegativeTypeLaw],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, NegativeTypeLaw]:
    """Transport conditional negative definiteness through composition."""
    del law, transform
    return SpaceLaw()


def restrict_law[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
    PropertyT: Law,
](
    law: SpaceLaw[TargetT, PropertyT],
    transform: InjectiveTransform[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, PropertyT]:
    """Preserve every declared law along an equality-injective restriction."""
    del law, transform
    return SpaceLaw()


def quotient_metric[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[SourceT, PseudometricLaw],
    transform: QuotientTransform[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
    witness: QuotientMetricWitness[SourceT, TargetT, MapKindT],
) -> SpaceLaw[TargetT, MetricLaw]:
    """Descend a pseudometric to a metric only with an exact-fibre witness."""
    del law, transform, witness
    return SpaceLaw()


def typed_jit[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Compile ``function`` while preserving its full static signature."""
    return jax.jit(function)


def typed_vmap[Input: Shape, Output: Shape, Batch: DimensionVar](
    function: Callable[[ShapedArray["Input"]], ShapedArray["Output"]],
) -> Callable[
    [ShapedArray[["Batch", *Elements["Input"]]]],
    ShapedArray[["Batch", *Elements["Output"]]],
]:
    """Map one array argument and prepend the batch axis to its array result.

    Arbitrary pytrees need their own leaf-shape contract; this adapter makes no
    same-signature promise for a transform that changes array ranks.
    """
    return jax.vmap(function)


_MAPPED_INPUT = "batch *input"
_MAPPED_OUTPUT = "batch *output"


def map_arrays[Input: Shape, Output: Shape, Batch: DimensionVar](
    function: Callable[[ShapedArray["Input"]], ShapedArray["Output"]],
    xs: Shaped[ShapedArray[["Batch", *Elements["Input"]]], _MAPPED_INPUT],
    chunk: int = 64,
) -> Shaped[ShapedArray[["Batch", *Elements["Output"]]], _MAPPED_OUTPUT]:
    """Map array slices in bounded chunks, retaining the prepended batch axis."""
    return jax.lax.map(function, xs, batch_size=chunk)
