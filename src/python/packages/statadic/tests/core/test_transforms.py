# jaxtyping numerical carrier contracts.
"""Runtime and static checks for mathematical and JAX transform contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from hypothesis import example, given
from statadic.core.domains import SpaceLaw
from statadic.core.transforms import TransformContract, typed_jit, typed_vmap

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(grid=jnp.arange(6.0).reshape(2, 3))
@given(grid=float_arrays((2, 3)))
def test_typed_transforms_preserve_runtime_carriers(grid: jnp.ndarray) -> None:
    """Compile and batch an ordinary typed result without changing its value."""

    def total(row: jnp.ndarray) -> jnp.ndarray:
        return row.sum()

    expected = grid.sum(axis=1)
    # Reassociating a three-term sum moves it by at most a few ulps of sum(|x|).
    tolerance = (
        4 * float(jnp.finfo(grid.dtype).eps) * float(jnp.abs(grid).sum(axis=1).max())
    )
    assert float(typed_jit(total)(grid[0])) == pytest.approx(
        float(expected[0]), abs=tolerance
    )
    assert typed_vmap(total)(grid).tolist() == pytest.approx(
        expected.tolist(), abs=tolerance
    )


@pytest.mark.unit
@pytest.mark.contract
def test_mathematical_transform_descriptors_are_static_zero_field_values() -> None:
    """Keep map and transported-law metadata out of numerical pytrees."""
    transform = TransformContract()
    law = SpaceLaw()

    assert transform == type(transform)()
    assert hash(transform) == hash(type(transform)())
    assert transform.__dataclass_fields__ == {}
    assert law.__dataclass_fields__ == {}


_STATIC_ACCEPTED_FIXTURE = """\
from __future__ import annotations
from jax import Array
import jax.numpy as jnp
from statadic.core.axioms import MetricLaw, NegativeTypeLaw, PseudometricLaw
from statadic.core.domains import (  # noqa: TC001
    DirectOrigin,
    EmpiricalMeasureSpace,
    L2UnitSphereSpace,
    NonzeroVectorCarrierSpace,
    NormalizationPullbackOrigin,
    PopulationMeasureSpace,
    PositiveRaySpace,
    ProjectiveSpace,
    PushforwardMeasureSpace,
    SpaceLaw,
)
from statadic.core.statistics import (  # noqa: TC001
    DecompositionOrigin,
    EstimateProvenance,
    RealizedEstimate,
    TotalDispersion,
    UStatisticEstimator,
    VStatisticEstimator,
    WithinDispersion,
)
from statadic.core.transforms import (
    EmpiricalPushforwardMap,
    InjectiveTransform,
    L2Normalization,
    L2NormalizationContract,
    NonzeroInputRequired,
    PopulationPushforwardMap,
    PositiveRayQuotient,
    ProjectiveQuotient,
    QuotientMetricWitness,
    QuotientTransform,
    RestrictionMap,
    TotalOnDeclaredSource,
    TransformContract,
    pullback_negative_type,
    pullback_pseudometric,
    quotient_metric,
    restrict_law,
    typed_jit,
    typed_vmap,
)

type Total = RealizedEstimate[
    Array, EstimateProvenance[TotalDispersion, VStatisticEstimator, DecompositionOrigin]
]
type Sibling = RealizedEstimate[
    Array, EstimateProvenance[WithinDispersion, UStatisticEstimator, DirectOrigin]
]


def cell(row: Array) -> Total:
    return RealizedEstimate(value=row.sum(), provenance=EstimateProvenance())


sphere_metric: SpaceLaw[L2UnitSphereSpace, MetricLaw] = SpaceLaw()
sphere_negative_type: SpaceLaw[L2UnitSphereSpace, NegativeTypeLaw] = SpaceLaw()
normalization: L2NormalizationContract = TransformContract()
raw_pseudometric: SpaceLaw[NonzeroVectorCarrierSpace, PseudometricLaw] = (
    pullback_pseudometric(sphere_metric, normalization)
)
raw_negative_type: SpaceLaw[NonzeroVectorCarrierSpace, NegativeTypeLaw] = (
    pullback_negative_type(sphere_negative_type, normalization)
)
ambient_metric: SpaceLaw[NonzeroVectorCarrierSpace, MetricLaw] = SpaceLaw()
sphere_inclusion: InjectiveTransform[
    L2UnitSphereSpace,
    NonzeroVectorCarrierSpace,
    RestrictionMap,
    TotalOnDeclaredSource,
    DirectOrigin,
] = InjectiveTransform()
restricted_metric: SpaceLaw[L2UnitSphereSpace, MetricLaw] = restrict_law(
    ambient_metric, sphere_inclusion
)
ray_quotient: QuotientTransform[
    NonzeroVectorCarrierSpace,
    PositiveRaySpace,
    PositiveRayQuotient,
    TotalOnDeclaredSource,
    NormalizationPullbackOrigin,
] = QuotientTransform()
projective_quotient: QuotientTransform[
    NonzeroVectorCarrierSpace,
    ProjectiveSpace,
    ProjectiveQuotient,
    TotalOnDeclaredSource,
    DirectOrigin,
] = QuotientTransform()
ray_witness = QuotientMetricWitness[
    NonzeroVectorCarrierSpace, PositiveRaySpace, PositiveRayQuotient
]()
projective_witness = QuotientMetricWitness[
    NonzeroVectorCarrierSpace, ProjectiveSpace, ProjectiveQuotient
]()
ray_metric: SpaceLaw[PositiveRaySpace, MetricLaw] = quotient_metric(
    raw_pseudometric, ray_quotient, ray_witness
)
empirical_pushforward: TransformContract[
    EmpiricalMeasureSpace,
    EmpiricalMeasureSpace,
    EmpiricalPushforwardMap,
    TotalOnDeclaredSource,
    NormalizationPullbackOrigin,
] = TransformContract()
population_pushforward: TransformContract[
    PopulationMeasureSpace,
    PushforwardMeasureSpace,
    PopulationPushforwardMap,
    TotalOnDeclaredSource,
    NormalizationPullbackOrigin,
] = TransformContract()


def normalization_factory(row: Array) -> L2NormalizationContract:
    del row
    return TransformContract()


grid = jnp.ones((2, 3))


def accepted(grid: Array) -> None:
    direct: Total = typed_jit(cell)(grid)
    del direct
"""

_STATIC_FIXTURE = (
    _STATIC_ACCEPTED_FIXTURE
    + """\


def rejected(grid: Array) -> None:
    jitted: Sibling = typed_jit(cell)(grid)  # E: is not assignable
    typed_vmap(cell)  # E: is not assignable
    del jitted


raw_metric: SpaceLaw[
    NonzeroVectorCarrierSpace, MetricLaw
] = pullback_pseudometric(sphere_metric, normalization)  # E: is not assignable


invalid_restriction = restrict_law(sphere_metric, normalization)  # E: is not assignable


wrong_quotient: QuotientTransform[
    NonzeroVectorCarrierSpace,
    PositiveRaySpace,
    PositiveRayQuotient,
    TotalOnDeclaredSource,
    NormalizationPullbackOrigin,
] = projective_quotient  # E: is not assignable


wrong_witness_metric = quotient_metric(
    raw_pseudometric, ray_quotient, projective_witness  # E: is not assignable
)


wrong_measure_map: TransformContract[
    PopulationMeasureSpace,
    PushforwardMeasureSpace,
    PopulationPushforwardMap,
    TotalOnDeclaredSource,
    NormalizationPullbackOrigin,
] = empirical_pushforward  # E: is not assignable


bad_jit_contract: TransformContract[
    NonzeroVectorCarrierSpace,
    ProjectiveSpace,
    ProjectiveQuotient,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
] = typed_jit(normalization_factory)(grid)  # E: is not assignable


typed_vmap(normalization_factory)  # E: is not assignable
"""
)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_typed_transforms_reject_a_sibling_brand(tmp_path: Path) -> None:
    """Keep every return-axis parameter visible to strict Pyrefly."""
    check_types(_STATIC_ACCEPTED_FIXTURE, "typed_transforms_accepted.py", tmp_path)
    check_types(_STATIC_FIXTURE, "typed_transforms.py", tmp_path)
