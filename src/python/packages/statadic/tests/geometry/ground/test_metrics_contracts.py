# standalone package-maintenance entry point
"""Space-, transform-, authority-, and estimand-indexed ground contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.axioms import (
    METRIC,
    PSEUDOMETRIC,
    Axioms,
)
from statadic.core.domains import DeclaredAuthority, GovernedReferenceAuthority
from statadic.core.matrices import SpaceDMat
from statadic.core.transforms import TransformDescriptor
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    PROJECTIVE_ANGULAR,
    RAY_ANGULAR,
    RAY_CHORD,
    SPHERE_ANGULAR,
    SPHERE_CHORD,
    checked_cloud_pdist,
    projective_angular_distance,
    sphere_angular_distance,
    sphere_chord_distance,
)
from statadic.sample.evidence import checked_unit_norm_clouds
from statadic.sample.packed import (
    pack,
)

from ..._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.core.axioms import (
        MetricLaw,
    )
    from statadic.geometry.ground.metrics import (
        GroundDistance,
    )

_STATIC_ACCEPTED_FIXTURE = """\
from __future__ import annotations

from jax import Array
from statadic.core.axioms import MetricLaw, PseudometricLaw
from statadic.core.domains import (  # noqa: TC001
    ClaimAuthority,
    CheckedEvidence,
    DirectOrigin,
    FiniteValues,
    GovernedReferenceAuthority,
    L2UnitSphereSpace,
    NonzeroVectorCarrierSpace,
    NormalizationPullbackOrigin,
    NonzeroNormalizationDomain,
    PositiveRaySpace,
    ProjectiveSpace,
    RawVector,
    RawVectorSpace,
    UnitNormSupport,
    UnitSphereValue,
)
from statadic.core.matrices import MatrixContract, SpaceDMat
from statadic.core.statistics import (  # noqa: TC001
    DiscoConstruction,
    EnergyConstruction,
    MmdConstruction,
    EstimateProvenance,
    RealizedEstimate,
    TransformedEstimand,
    VStatisticEstimator,
)
from statadic.core.domains import EnergyOrigin
from statadic.core.transforms import (
    IdentityMap,
    L2Normalization,
    NonzeroInputRequired,
    PositiveRayQuotient,
    TotalOnDeclaredSource,
    TransformContract,
    TransformDescriptor,
)
from statadic.geometry.ground.metrics import (  # noqa: TC001
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    PROJECTIVE_ANGULAR,
    RAY_ANGULAR,
    RAY_CHORD,
    SPHERE_ANGULAR,
    SPHERE_CHORD,
    EuclideanLaw,
    GroundDistance,
    checked_cloud_pdist,
)
from statadic.sample.evidence import CheckedClouds, Unchecked

type SphereDirect = TransformContract[
    L2UnitSphereSpace,
    L2UnitSphereSpace,
    IdentityMap,
    TotalOnDeclaredSource,
    DirectOrigin,
]
type Normalization = TransformContract[
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    L2Normalization,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
]
type RawDirect = TransformContract[
    RawVectorSpace,
    RawVectorSpace,
    IdentityMap,
    TotalOnDeclaredSource,
    DirectOrigin,
]
type PositiveRayTransform = TransformContract[
    NonzeroVectorCarrierSpace,
    PositiveRaySpace,
    PositiveRayQuotient,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
]

def needs_raw_metric(
    strategy: GroundDistance[
        MetricLaw,
        RawVectorSpace,
        TransformDescriptor,
        ClaimAuthority,
        FiniteValues,
    ],
) -> None: ...
def needs_sphere_metric(
    strategy: GroundDistance[
        MetricLaw,
        L2UnitSphereSpace,
        TransformDescriptor,
        ClaimAuthority,
        UnitNormSupport,
    ],
) -> None: ...
def needs_nonzero_pseudometric(
    strategy: GroundDistance[
        PseudometricLaw,
        NonzeroVectorCarrierSpace,
        TransformDescriptor,
        ClaimAuthority,
        NonzeroNormalizationDomain,
    ],
) -> None: ...
def needs_projective_metric(
    strategy: GroundDistance[
        MetricLaw,
        ProjectiveSpace,
        TransformDescriptor,
        ClaimAuthority,
        NonzeroNormalizationDomain,
    ],
) -> None: ...
def needs_positive_ray_metric(
    strategy: GroundDistance[
        MetricLaw,
        PositiveRaySpace,
        PositiveRayTransform,
        ClaimAuthority,
        NonzeroNormalizationDomain,
    ],
) -> None: ...
def needs_sphere_direct(
    strategy: GroundDistance[
        MetricLaw,
        L2UnitSphereSpace,
        SphereDirect,
        GovernedReferenceAuthority,
        UnitNormSupport,
    ],
) -> None: ...
def needs_normalization_pullback(
    strategy: GroundDistance[
        PseudometricLaw,
        NonzeroVectorCarrierSpace,
        Normalization,
        GovernedReferenceAuthority,
        NonzeroNormalizationDomain,
    ],
) -> None: ...

def accepted() -> None:
    needs_raw_metric(EUCLIDEAN)
    needs_sphere_metric(SPHERE_ANGULAR)
    needs_sphere_metric(SPHERE_CHORD)
    needs_nonzero_pseudometric(ANGULAR)
    needs_nonzero_pseudometric(NORMALIZED_EUCLIDEAN)
    needs_projective_metric(PROJECTIVE_ANGULAR)
    needs_positive_ray_metric(RAY_ANGULAR)
    needs_positive_ray_metric(RAY_CHORD)
    needs_sphere_direct(SPHERE_CHORD)
    needs_normalization_pullback(NORMALIZED_EUCLIDEAN)

type CheckedSphereChord = MatrixContract[
    EuclideanLaw,
    SphereDirect,
    GovernedReferenceAuthority,
    UnitNormSupport,
    CheckedEvidence,
]

def matrix_propagation(
    unit: CheckedClouds[
        UnitSphereValue,
        UnitNormSupport,
        Unchecked,
        CheckedEvidence,
    ],
) -> None:
    exact: SpaceDMat[L2UnitSphereSpace, CheckedSphereChord] = checked_cloud_pdist(
        unit, SPHERE_CHORD
    )
    del exact

type EnergyPushforward = TransformedEstimand[
    EnergyConstruction,
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    Normalization,
]
type MmdPushforward = TransformedEstimand[
    MmdConstruction,
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    Normalization,
]
type DiscoPushforward = TransformedEstimand[
    DiscoConstruction,
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    Normalization,
]
type DirectEnergy = TransformedEstimand[
    EnergyConstruction,
    RawVectorSpace,
    RawVectorSpace,
    RawDirect,
]

def needs_energy_pushforward(
    result: RealizedEstimate[
        Array, EstimateProvenance[EnergyPushforward, VStatisticEstimator, EnergyOrigin]
    ],
) -> None: ...
def needs_direct_energy(
    result: RealizedEstimate[
        Array, EstimateProvenance[DirectEnergy, VStatisticEstimator, EnergyOrigin]
    ],
) -> None: ...

def estimand_acceptance(
    energy: RealizedEstimate[
        Array, EstimateProvenance[EnergyPushforward, VStatisticEstimator, EnergyOrigin]
    ],
    direct: RealizedEstimate[
        Array, EstimateProvenance[DirectEnergy, VStatisticEstimator, EnergyOrigin]
    ],
) -> None:
    needs_direct_energy(direct)
    needs_energy_pushforward(energy)
"""

_STATIC_FIXTURE = (
    _STATIC_ACCEPTED_FIXTURE
    + """\
def rejected() -> None:
    needs_raw_metric(ANGULAR)  # E: is not assignable
    needs_sphere_metric(ANGULAR)  # E: is not assignable
    needs_nonzero_pseudometric(COSINE)  # E: is not assignable
    needs_projective_metric(SPHERE_ANGULAR)  # E: is not assignable
    needs_sphere_direct(NORMALIZED_EUCLIDEAN)  # E: is not assignable
    needs_normalization_pullback(SPHERE_CHORD)  # E: is not assignable
    needs_positive_ray_metric(PROJECTIVE_ANGULAR)  # E: is not assignable

def matrix_rejections(
    exact: SpaceDMat[L2UnitSphereSpace, CheckedSphereChord],
    finite: CheckedClouds[RawVector, FiniteValues, Unchecked, CheckedEvidence],
) -> None:
    wrong_space: SpaceDMat[NonzeroVectorCarrierSpace, CheckedSphereChord] = (
        exact  # E: is not assignable
    )
    insufficient_evidence: SpaceDMat[L2UnitSphereSpace, CheckedSphereChord] = (
        checked_cloud_pdist(finite, SPHERE_CHORD)  # E: is not assignable
    )
    del wrong_space, insufficient_evidence

def estimand_rejections(
    energy: RealizedEstimate[
        Array, EstimateProvenance[EnergyPushforward, VStatisticEstimator, EnergyOrigin]
    ],
    mmd: RealizedEstimate[
        Array, EstimateProvenance[MmdPushforward, VStatisticEstimator, EnergyOrigin]
    ],
    disco: RealizedEstimate[
        Array, EstimateProvenance[DiscoPushforward, VStatisticEstimator, EnergyOrigin]
    ],
) -> None:
    needs_direct_energy(energy)  # E: is not assignable
    needs_energy_pushforward(mmd)  # E: is not assignable
    needs_energy_pushforward(disco)  # E: is not assignable
"""
)


def _primitive_metric_battery(values: np.ndarray) -> None:
    """Check metric primitives independently on one admissible finite carrier."""
    tolerance = 1e-6
    assert np.all(values >= -tolerance)
    assert np.allclose(np.diag(values), 0.0, atol=tolerance)
    assert np.allclose(values, values.T, atol=tolerance)
    assert np.all(values[np.triu_indices(values.shape[0], 1)] > tolerance)
    for left in range(values.shape[0]):
        for middle in range(values.shape[0]):
            for right in range(values.shape[0]):
                assert values[left, right] <= (
                    values[left, middle] + values[middle, right] + tolerance
                )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("strategy", [SPHERE_ANGULAR, SPHERE_CHORD])
def test_direct_sphere_metrics_declare_governed_authority(
    strategy: GroundDistance[MetricLaw],
) -> None:
    """Direct sphere laws retain their governed mathematical authority."""
    assert strategy.law.axioms == METRIC
    assert isinstance(strategy.authority, GovernedReferenceAuthority)
    assert isinstance(strategy.transform, TransformDescriptor)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("strategy", [RAY_ANGULAR, RAY_CHORD, PROJECTIVE_ANGULAR])
def test_quotient_metrics_do_not_overstate_their_authority(
    strategy: GroundDistance[MetricLaw],
) -> None:
    """Unformalized quotient laws remain explicit type declarations."""
    assert strategy.law.axioms == METRIC
    assert isinstance(strategy.authority, DeclaredAuthority)
    assert isinstance(strategy.transform, TransformDescriptor)


@pytest.mark.unit
@pytest.mark.property
def test_direct_sphere_metrics_pass_primitive_laws_on_unit_rows() -> None:
    """Direct angular and chord formulas are metrics on checked sphere points."""
    points = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    for distance in (sphere_angular_distance, sphere_chord_distance):
        _primitive_metric_battery(np.asarray(distance(points, points)))


@pytest.mark.unit
@pytest.mark.property
def test_projective_angular_uses_projective_not_positive_ray_equality() -> None:
    """Antipodal representatives coincide; distinct projective classes separate."""
    root_half = np.sqrt(0.5)
    representatives = jnp.array([[1.0, 0.0], [0.0, 1.0], [root_half, root_half]])
    _primitive_metric_battery(
        np.asarray(projective_angular_distance(representatives, representatives))
    )
    antipodes = projective_angular_distance(
        jnp.array([[1.0, 0.0]]),
        jnp.array([[-2.0, 0.0]]),
    )
    assert float(antipodes[0, 0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.component
@pytest.mark.property
def test_raw_normalization_strategies_remain_weaker_than_metrics() -> None:
    """Raw angular/chord pullbacks are pseudometrics; cosine also lacks triangle."""
    assert ANGULAR.law.axioms == PSEUDOMETRIC
    assert NORMALIZED_EUCLIDEAN.law.axioms == PSEUDOMETRIC
    assert not bool(COSINE.law.axioms & Axioms.TRIANGLE)
    assert EUCLIDEAN.law.axioms == METRIC


@pytest.mark.component
@pytest.mark.contract
def test_checked_cloud_matrix_retains_every_semantic_axis() -> None:
    """The matrix has one array leaf while semantic facts stay static."""
    checked = checked_unit_norm_clouds(pack([jnp.eye(2)]))
    matrix = checked_cloud_pdist(checked, SPHERE_CHORD)
    assert isinstance(matrix, SpaceDMat)
    assert len(jax.tree_util.tree_leaves(matrix)) == 1
    np.testing.assert_allclose(
        matrix.values,
        sphere_chord_distance(jnp.eye(2), jnp.eye(2)),
    )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_strategy_space_transform_and_estimand_rejections_are_static(
    tmp_path: Path,
) -> None:
    """Pyrefly rejects wrong spaces, maps, laws, and transformed estimands."""
    check_types(_STATIC_ACCEPTED_FIXTURE, "ground_space_accepted.py", tmp_path)
    check_types(_STATIC_FIXTURE, "ground_space_contracts.py", tmp_path)
