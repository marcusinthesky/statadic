# jaxtyping shape strings are runtime contracts.
"""``core/domains.py`` — the carrier lattice, as pyrefly errors rather than prose.

The mathematics this pins is not stylistic. One formula,
``d(x, y) = ‖N(x) - N(y)‖``, is a metric on checked unit-sphere values, a
pseudometric on raw nonzero vectors, a metric on the positive-ray quotient, and
not a metric on real projective space. Four carriers, four separation facts, and
a consumer that confuses them draws a false identification conclusion with no
runtime symptom — every distance it reports is still finite, symmetric, and
nonnegative.

So the negative fixtures below are the actual deliverable. As in
``tests/core/test_typing.py``, each is written to ``tmp_path`` and checked in a
subprocess: a deliberately-wrong module cannot live in the tree, because
``just python::typecheck`` passes ``apps packages`` and gates on zero errors.

Native Pyrefly expectations bind each rejection to its source location and
reject unexpected diagnostics, including import failures. Accepted fixtures
also run independently without any expected errors.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING

import pytest
from statadic.core.domains import (
    AlmostSureNonzeroSupport,
    FiniteMoment,
    FiniteValues,
    L1UnitNormSupport,
    L1UnitSphereValue,
    L2UnitSphereValue,
    LInfUnitNormSupport,
    LInfUnitSphereValue,
    MathematicalSpace,
    NonzeroNormalizationDomain,
    NonzeroPooledMean,
    NonzeroVector,
    OrientedRayClass,
    PooledPrototype,
    PushforwardLaw,
    RawVector,
    Separation,
    UnitNormSupport,
    UnitSphereValue,
)

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


_CARRIER_ACCEPTED_FIXTURE = '''\
"""T55.3: object domains, separation transport, and construction origins.

Deliberately-widening fixture; never a checked-in `.py` file. See the module
docstring of ``tests/core/test_domains.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from statadic.core.axioms import PseudometricLaw
from statadic.core.domains import (  # noqa: TC001
    ConstructionOrigin,
    DirectOrigin,
    Document,
    DocumentRepresentation,
    DomainLaw,
    DomainT,
    EmpiricalDistribution,
    EnergyOrigin,
    FeatureMap,
    InjectiveFeatureMap,
    NonzeroVector,
    NormalizationPullbackOrigin,
    OrientedRayClass,
    OriginT,
    PopulationLaw,
    PushforwardLaw,
    RawVector,
    Separation,
    UnitSphereValue,
    pullback_separation,
)


@dataclass(frozen=True, slots=True)
class Carrier(Generic[DomainT]):
    """Stand-in for the branded containers `core/matrices.py` will own."""


@dataclass(frozen=True, slots=True)
class Built(Generic[OriginT]):
    """Stand-in for a value branded by how it was constructed."""


def needs_unit_carrier(values: Carrier[UnitSphereValue]) -> None: ...
def needs_ray_carrier(values: Carrier[OrientedRayClass]) -> None: ...
def needs_nonzero_carrier(values: Carrier[NonzeroVector]) -> None: ...
def needs_population_carrier(values: Carrier[PopulationLaw]) -> None: ...
def needs_raw_separation(fact: Separation[RawVector]) -> None: ...
def needs_document_separation(fact: Separation[Document]) -> None: ...
def needs_population_separation(fact: Separation[PopulationLaw]) -> None: ...
def needs_raw_domain_law(law: DomainLaw[RawVector, PseudometricLaw]) -> None: ...
def needs_any_origin(value: Built[ConstructionOrigin]) -> None: ...
def needs_direct_origin(value: Built[DirectOrigin]) -> None: ...


def accepted(
    unit: Carrier[UnitSphereValue],
    pushforward: Carrier[PushforwardLaw],
    ray_separation: Separation[OrientedRayClass],
    raw_separation: Separation[RawVector],
    source_separation: Separation[PopulationLaw],
    energy: Built[EnergyOrigin],
    pullback: Built[NormalizationPullbackOrigin],
    representation: Separation[DocumentRepresentation],
    injective: InjectiveFeatureMap[Document, DocumentRepresentation],
) -> None:
    # A checked unit-sphere value is a representative of a positive-ray class.
    needs_ray_carrier(unit)
    # ...and is still a finite nonzero vector.
    needs_nonzero_carrier(unit)
    # A pushforward N#P is a population law.
    needs_population_carrier(pushforward)
    # Separating a coarser equality is the stronger fact.
    needs_raw_separation(raw_separation)
    needs_population_separation(source_separation)
    # A specific origin satisfies a generic requirement, in both steps.
    needs_any_origin(energy)
    needs_direct_origin(pullback)
    needs_any_origin(pullback)
    # Document separation is reachable, but only through the injectivity witness.
    needs_document_separation(pullback_separation(representation, injective))
    del ray_separation
'''

_CARRIER_FIXTURE = (
    _CARRIER_ACCEPTED_FIXTURE
    + """\
def rejected(
    raw: Carrier[RawVector],
    empirical: Carrier[EmpiricalDistribution],
    unit_separation: Separation[UnitSphereValue],
    ray_separation: Separation[OrientedRayClass],
    representation: Separation[DocumentRepresentation],
    pushforward_separation: Separation[PushforwardLaw],
    normalizing_law: DomainLaw[NonzeroVector, PseudometricLaw],
    direct: Built[DirectOrigin],
    plain: FeatureMap[Document, DocumentRepresentation],
) -> None:
    # raw vectors are not checked unit values
    needs_unit_carrier(raw)  # E: is not assignable
    # a sample is not a population law
    needs_population_carrier(empirical)  # E: is not assignable
    # chord pullback is a pseudometric
    needs_raw_separation(unit_separation)  # E: is not assignable
    # no injectivity witness
    needs_document_separation(representation)  # E: is not assignable
    needs_population_separation(
        pushforward_separation  # E: is not assignable
    )  # ERROR: N#P = N#Q does not give P = Q
    # N is undefined on a zero row
    needs_raw_domain_law(normalizing_law)  # E: is not assignable
    needs_direct_origin(direct)
    del ray_separation, plain
"""
)

_INJECTIVITY_FIXTURE = '''\
"""T55.3: separation transport requires a declared *injective* feature map."""

from __future__ import annotations

from statadic.core.domains import (  # noqa: TC001
    Document,
    DocumentRepresentation,
    FeatureMap,
    Separation,
    pullback_separation,
)


def transport(
    representation: Separation[DocumentRepresentation],
    plain: FeatureMap[Document, DocumentRepresentation],
) -> Separation[Document]:
    # not injective
    return pullback_separation(representation, plain)  # E: is not assignable
'''

_SPACE_ACCEPTED_FIXTURE = '''\
"""T55.6: carrier, equality, and algebraic structure are independent axes."""

from __future__ import annotations

from statadic.core.domains import (  # noqa: TC001
    L1UnitSphereSpace,
    L2UnitSphereSpace,
    PositiveRaySpace,
    ProjectiveSpace,
    RawVectorSpace,
)


def needs_l2_sphere(space: L2UnitSphereSpace) -> None: ...
def needs_vector_space(space: RawVectorSpace) -> None: ...
def needs_positive_ray(space: PositiveRaySpace) -> None: ...
def needs_projective(space: ProjectiveSpace) -> None: ...


def accepted(
    raw: RawVectorSpace,
    l2: L2UnitSphereSpace,
    ray: PositiveRaySpace,
    projective: ProjectiveSpace,
) -> None:
    needs_vector_space(raw)
    needs_l2_sphere(l2)
    needs_positive_ray(ray)
    needs_projective(projective)
'''

_SPACE_FIXTURE = (
    _SPACE_ACCEPTED_FIXTURE
    + """\
def rejected(
    l1: L1UnitSphereSpace,
    l2: L2UnitSphereSpace,
    ray: PositiveRaySpace,
    projective: ProjectiveSpace,
) -> None:
    # L1 evidence is not L2 evidence
    needs_l2_sphere(l1)  # E: is not assignable
    # a sphere is not a vector space
    needs_vector_space(l2)  # E: is not assignable
    # positive rays do not identify antipodes
    needs_projective(ray)  # E: is not assignable
    # projective equality is coarser
    needs_positive_ray(projective)  # E: is not assignable
"""
)

_EVIDENCE_ACCEPTED_FIXTURE = '''\
"""T55.3: value evidence implies only what a checked door actually measured."""

from __future__ import annotations

from statadic.core.domains import (  # noqa: TC001
    AlmostSureNonzeroSupport,
    AlphaMomentOrder,
    FiniteMoment,
    FiniteValues,
    FirstMomentOrder,
    L1UnitNormSupport,
    NonzeroNormalizationDomain,
    NonzeroPooledMean,
    UnitNormSupport,
)


def needs_finite(evidence: FiniteValues) -> None: ...
def needs_normalization_domain(evidence: NonzeroNormalizationDomain) -> None: ...
def needs_unit_norm(evidence: UnitNormSupport) -> None: ...
def needs_pooled_mean(evidence: NonzeroPooledMean) -> None: ...
def needs_population_support(evidence: AlmostSureNonzeroSupport) -> None: ...
def needs_first_moment(evidence: FiniteMoment[FirstMomentOrder]) -> None: ...


def accepted(
    unit: UnitNormSupport,
    pooled: NonzeroPooledMean,
    first_moment: FiniteMoment[FirstMomentOrder],
) -> None:
    needs_normalization_domain(unit)
    needs_finite(unit)
    needs_finite(pooled)
    needs_first_moment(first_moment)
'''

_EVIDENCE_FIXTURE = (
    _EVIDENCE_ACCEPTED_FIXTURE
    + """\
def rejected(
    l1_unit: L1UnitNormSupport,
    unit: UnitNormSupport,
    finite: FiniteValues,
    normalization: NonzeroNormalizationDomain,
    alpha_moment: FiniteMoment[AlphaMomentOrder],
) -> None:
    # L1 unit norm is not L2 unit norm
    needs_unit_norm(l1_unit)  # E: is not assignable
    # The single most important negative constraint: an antipodal unit cloud
    # averages to exactly zero.
    # unit rows do not give a nonzero pooled mean
    needs_pooled_mean(unit)  # E: is not assignable
    # finiteness is not unit norm
    needs_unit_norm(finite)  # E: is not assignable
    needs_population_support(
        normalization  # E: is not assignable
    )  # ERROR: checking observed rows is not an almost-sure population policy
    # moment orders do not interconvert
    needs_first_moment(alpha_moment)  # E: is not assignable
"""
)

_JOIN_FIXTURE = '''\
"""T55.3: the unsound unit-norm/pooled-mean join cannot be reintroduced."""

from __future__ import annotations

from statadic.core.domains import NonzeroPooledMean, UnitNormSupport


# both are final
class BothFacts(
    UnitNormSupport,  # E: Cannot extend final class
    NonzeroPooledMean,  # E: Cannot extend final class
):
    pass
'''


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_carrier_domains_and_origins_are_static(tmp_path: Path) -> None:
    """Pyrefly accepts every sound widening and rejects six unsound ones.

    The accepted half matters as much as the rejected half. Invariant phantom
    parameters would also reject ``needs_ray_carrier(unit)``, which is the
    legal direction and the reason the axes carry declared variance at all.
    """
    check_types(_CARRIER_ACCEPTED_FIXTURE, "carriers_accepted.py", tmp_path)
    check_types(_CARRIER_FIXTURE, "carriers.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_document_separation_requires_an_injectivity_witness(tmp_path: Path) -> None:
    """An ordinary feature map cannot transport separation back to documents.

    Pulling a representation distance back to documents is only a pseudometric
    unless the encoder is injective, so ``pullback_separation`` accepts nothing
    weaker than ``InjectiveFeatureMap``.
    """
    check_types(_INJECTIVITY_FIXTURE, "injectivity.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_space_axes_reject_norm_structure_and_quotient_confusion(
    tmp_path: Path,
) -> None:
    """Space descriptors make four array-representation confusions static errors."""
    check_types(_SPACE_ACCEPTED_FIXTURE, "spaces_accepted.py", tmp_path)
    check_types(_SPACE_FIXTURE, "spaces.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_value_evidence_implications_are_static(tmp_path: Path) -> None:
    """Unit rows never widen into a nonzero pooled mean, and moments stay apart."""
    check_types(_EVIDENCE_ACCEPTED_FIXTURE, "evidence_accepted.py", tmp_path)
    check_types(_EVIDENCE_FIXTURE, "evidence.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_the_unsound_evidence_join_is_unrepresentable(tmp_path: Path) -> None:
    """``@final`` is what makes constraint 3 unrepresentable rather than unstated.

    Nominal sibling evidence classes are only siblings while nobody writes a
    "fully checked" subclass of both. A downstream eager door in
    ``sample/evidence.py`` would find that convenient; ``@final`` turns it into
    a static error in *their* file rather than a silent hole in this lattice.
    """
    check_types(_JOIN_FIXTURE, "join.py", tmp_path)


# --- runtime shape of the markers ---------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_domain_markers_are_leafless_phantoms() -> None:
    """Domain markers are plain empty classes; nothing reaches an array tree.

    ``UnitSphereValue`` inherits from two parents, which is why the markers are
    bare classes rather than slotted dataclasses: multiple inheritance over two
    ``slots=True`` dataclasses is a layout conflict at import time.
    """
    assert UnitSphereValue.__mro__.index(NonzeroVector) < UnitSphereValue.__mro__.index(
        OrientedRayClass
    )
    assert issubclass(UnitSphereValue, L2UnitSphereValue)
    assert not issubclass(L1UnitSphereValue, L2UnitSphereValue)
    assert not issubclass(LInfUnitSphereValue, L2UnitSphereValue)
    assert issubclass(UnitSphereValue, (NonzeroVector, OrientedRayClass))
    assert issubclass(PooledPrototype, NonzeroVector)
    assert not issubclass(UnitSphereValue, PooledPrototype)
    assert not issubclass(OrientedRayClass, RawVector)
    assert not hasattr(UnitSphereValue, "__slots__")
    assert not inspect.get_annotations(UnitSphereValue)
    assert not dataclasses.is_dataclass(UnitSphereValue)


@pytest.mark.unit
@pytest.mark.contract
def test_pushforward_is_a_population_law_but_a_sample_is_not() -> None:
    """The measure side of the lattice, checked nominally as well as statically."""
    from statadic.core.domains import (  # noqa: PLC0415
        EmpiricalDistribution,
        MeasureDomain,
        PopulationLaw,
    )

    assert issubclass(PushforwardLaw, PopulationLaw)
    assert issubclass(EmpiricalDistribution, MeasureDomain)
    assert not issubclass(EmpiricalDistribution, PopulationLaw)


@pytest.mark.unit
@pytest.mark.contract
def test_evidence_markers_are_frozen_and_zero_field() -> None:
    """Evidence is static metadata: hashable, immutable, and carries no arrays."""
    evidence: tuple[FiniteValues, ...] = (
        FiniteValues(),
        NonzeroNormalizationDomain(),
        L1UnitNormSupport(),
        UnitNormSupport(),
        LInfUnitNormSupport(),
        NonzeroPooledMean(),
    )
    for marker in (*evidence, AlmostSureNonzeroSupport(), FiniteMoment()):
        assert hash(marker) is not None
        assert not getattr(marker, "__dict__", {})
    assert UnitNormSupport() == UnitNormSupport()
    assert not isinstance(UnitNormSupport(), NonzeroPooledMean)


@pytest.mark.unit
@pytest.mark.contract
def test_mathematical_space_descriptors_are_zero_field() -> None:
    """Space descriptions remain static metadata rather than array leaves."""
    descriptor = MathematicalSpace()
    assert dataclasses.fields(descriptor) == ()
    assert hash(descriptor) is not None


@pytest.mark.unit
@pytest.mark.contract
def test_separation_transport_carries_no_array_state() -> None:
    """``pullback_separation`` moves a *type*; the value is an empty static marker.

    The claim is not that JAX sees no leaf — an unregistered object *is* a leaf
    if you hand it to a transform. It is that these markers have no fields, so
    they can only ever travel as hashable ``static_argnames`` metadata and can
    never smuggle an array into a compiled tree.
    """
    import dataclasses  # only needed for this assertion  # noqa: PLC0415

    from statadic.core.domains import (  # noqa: PLC0415
        InjectiveFeatureMap,
        pullback_separation,
    )

    transported = pullback_separation(Separation(), InjectiveFeatureMap())
    assert dataclasses.fields(transported) == ()
    assert dataclasses.fields(InjectiveFeatureMap()) == ()
    assert hash(transported) == hash(Separation())


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "marker",
    [UnitNormSupport, NonzeroPooledMean],
)
def test_the_antipodal_pair_stays_final(marker: type) -> None:
    """The two classes whose join would destroy constraint 3 are ``@final``."""
    assert getattr(marker, "__final__", False) is True
