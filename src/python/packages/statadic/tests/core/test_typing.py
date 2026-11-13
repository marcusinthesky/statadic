"""Canonical matrix pytrees, structural law variance and static vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxtyping
import pytest
from statadic.core.axioms import MetricLaw
from statadic.core.domains import DirectOrigin, RawVector
from statadic.core.matrices import DMat, MatrixProvenance, unsafe_assume_dmat

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


type MetricMatrix = DMat[RawVector, MatrixProvenance[MetricLaw, DirectOrigin]]


def _matrix_sum(matrix: MetricMatrix) -> jax.Array:
    """Project the numeric leaf without erasing the declared matrix axes."""
    return matrix.values.sum()


@pytest.mark.component
@pytest.mark.contract
def test_dmat_is_a_transparent_pytree() -> None:
    """Canonical matrices retain one numerical leaf through JIT."""
    d: MetricMatrix = unsafe_assume_dmat(jnp.eye(4))
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(d)] == [(4, 4)]
    assert float(jax.jit(_matrix_sum)(d)) == pytest.approx(4)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.needs_import_hook
@pytest.mark.parametrize("shape", [(3, 4), (5,), (2, 3, 4)])
def test_dmat_shape_is_enforced_at_runtime(shape: tuple[int, ...]) -> None:
    """Only a rank-two distance matrix enters the canonical constructor."""
    with pytest.raises(jaxtyping.TypeCheckError):
        unsafe_assume_dmat(jnp.ones(shape))


@pytest.mark.component
@pytest.mark.contract
def test_dmat_survives_vmap_unflatten() -> None:
    """Batch transforms preserve the matrix wrapper and numerical leaves."""
    stack = jnp.stack([jnp.eye(3), 2 * jnp.eye(3)])
    batched: MetricMatrix = jax.vmap(unsafe_assume_dmat)(stack)
    assert batched.values.shape == (2, 3, 3)
    assert jax.vmap(_matrix_sum)(batched).tolist() == pytest.approx([3, 6])


_WIDENING_ACCEPTED_FIXTURE = """
from statadic.core.axioms import MetricLaw, NegativeTypeLaw, PremetricLaw
from statadic.core.domains import DirectOrigin, RawVector
from statadic.core.matrices import DMat, MatrixProvenance

def needs_negative_type(
    d: DMat[RawVector, MatrixProvenance[NegativeTypeLaw, DirectOrigin]],
) -> None: ...
def accepts_premetric(
    d: DMat[RawVector, MatrixProvenance[PremetricLaw, DirectOrigin]],
) -> None: ...
def narrow(metric: DMat[RawVector, MatrixProvenance[MetricLaw, DirectOrigin]]) -> None:
    accepts_premetric(metric)
"""

_WIDENING_FIXTURE = (
    _WIDENING_ACCEPTED_FIXTURE
    + """
def widen(semi: DMat[RawVector, MatrixProvenance[PremetricLaw, DirectOrigin]]) -> None:
    needs_negative_type(semi)  # E: is not assignable
"""
)

_LAW_CAPABILITY_ACCEPTED_FIXTURE = '''\
"""T55.1: orthogonal structural laws widen only in sound directions."""

from __future__ import annotations

from typing import Generic, Protocol

from statadic.core.axioms import (  # noqa: TC001
    DivergenceLaw,
    MetricLaw,
    NegativeTypeLaw,
    PseudometricLaw,
    QuasimetricLaw,
    SemimetricLaw,
    StrongNegativeTypeLaw,
    Symmetric,
)
from statadic.core.typing import LawT


class Evidence(Generic[LawT]):
    pass


class SymmetricNegativeType(Symmetric, NegativeTypeLaw, Protocol):
    pass


class StrongNegativeMetric(MetricLaw, StrongNegativeTypeLaw, Protocol):
    pass


def needs_metric(law: Evidence[MetricLaw]) -> None: ...
def needs_pseudometric(law: Evidence[PseudometricLaw]) -> None: ...
def needs_quasimetric(law: Evidence[QuasimetricLaw]) -> None: ...
def needs_semimetric(law: Evidence[SemimetricLaw]) -> None: ...
def needs_divergence(law: Evidence[DivergenceLaw]) -> None: ...
def needs_negative_type(law: Evidence[NegativeTypeLaw]) -> None: ...
def needs_strong_negative_type(law: Evidence[StrongNegativeTypeLaw]) -> None: ...
def needs_symmetric_negative(law: Evidence[SymmetricNegativeType]) -> None: ...


def accepted(
    metric: Evidence[MetricLaw],
    strong: Evidence[StrongNegativeTypeLaw],
    negative_nonmetric: Evidence[SymmetricNegativeType],
    combined: Evidence[StrongNegativeMetric],
) -> None:
    needs_pseudometric(metric)
    needs_quasimetric(metric)
    needs_semimetric(metric)
    needs_divergence(metric)
    needs_negative_type(strong)
    needs_negative_type(negative_nonmetric)
    needs_metric(combined)
    needs_negative_type(combined)
    needs_strong_negative_type(combined)
    needs_symmetric_negative(combined)
'''

_LAW_CAPABILITY_FIXTURE = (
    _LAW_CAPABILITY_ACCEPTED_FIXTURE
    + """\
def rejected(
    pseudometric: Evidence[PseudometricLaw],
    quasimetric: Evidence[QuasimetricLaw],
    semimetric: Evidence[SemimetricLaw],
    divergence: Evidence[DivergenceLaw],
    negative: Evidence[NegativeTypeLaw],
    metric: Evidence[MetricLaw],
) -> None:
    # no separation
    needs_metric(pseudometric)  # E: is not assignable
    # no symmetry
    needs_metric(quasimetric)  # E: is not assignable
    # no triangle inequality
    needs_metric(semimetric)  # E: is not assignable
    # no symmetry or triangle inequality
    needs_metric(divergence)  # E: is not assignable
    # negative type need not be a metric
    needs_metric(negative)  # E: is not assignable
    # reverse property flow
    needs_strong_negative_type(negative)  # E: is not assignable
    # negative type is orthogonal
    needs_negative_type(metric)  # E: is not assignable
    # composed requirement is stronger
    needs_symmetric_negative(metric)  # E: is not assignable
"""
)

_MMD_BRAND_ACCEPTED_FIXTURE = '''\
"""T55.4: declared kernel capabilities propagate into rooted MMD laws."""

from __future__ import annotations

import jax.numpy as jnp
from statadic.discrepancy.mmd.energy import UNIT_EXPONENT, energy_geometry
from statadic.discrepancy.mmd import (
    LINEAR_KERNEL,
    distance_induced_kernel,
    mmd_matrix,
    rbf_kernel,
)
from statadic.core.domains import ConstructionOrigin, ObjectDomain
from statadic.core.matrices import (
    MatrixProvenance, DMat, SquaredDistanceNegativeTypeLaw,
)
from statadic.geometry.ground.metrics import ANGULAR
from statadic.geometry.ground.similarities import RBFKernel
from statadic.sample.packed import pack


def squared_negative_type_sink[D: ObjectDomain, O: ConstructionOrigin](
    distances: DMat[D, MatrixProvenance[SquaredDistanceNegativeTypeLaw, O]],
) -> None:
    """Sink for classical scaling's declared squared-negative-type law."""


def accepted() -> None:
    clouds = pack([jnp.ones((3, 2)), 2.0 * jnp.ones((3, 2))])
    characteristic = mmd_matrix(clouds, kernel=rbf_kernel(1.0))
    squared_negative_type_sink(characteristic)  # OK: characteristic Hilbertian metric
    dynamic = mmd_matrix(clouds, kernel=RBFKernel(jnp.asarray(1.0)))
    squared_negative_type_sink(dynamic)  # OK: dynamic parameters keep the law
    angular = mmd_matrix(
        clouds,
        kernel=distance_induced_kernel(energy_geometry(ANGULAR, UNIT_EXPONENT)),
    )
    squared_negative_type_sink(angular)  # OK: pseudometrics need not separate
'''

_MMD_BRAND_FIXTURE = (
    _MMD_BRAND_ACCEPTED_FIXTURE
    + """\


def rejected() -> None:
    clouds = pack([jnp.ones((3, 2)), 2.0 * jnp.ones((3, 2))])
    mmd_matrix(clouds, kernel=lambda x, y: x @ y.T)  # E: No matching overload
    mmd_matrix(clouds, kernel=LINEAR_KERNEL, unbiased=True)  # E: No matching overload
"""
)

_TYPED_DISTANCE_ACCEPTED_FIXTURE = '''\
"""T55.2: signatures, exact laws, config boundaries, and energy regimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp

from statadic.core.axioms import METRIC, Axioms, LawProperties
from statadic.core.domains import FiniteValues
from statadic.core.typing import LawfulCallable
from statadic.discrepancy.mmd.energy import (  # noqa: E402
    ANOVA_EXPONENT,
    UNIT_EXPONENT,
    DistributionEnergyLaw,
    EnergyGeometry,  # noqa: TC001
    MeanEnergyLaw,
    SeparatingDistributionEnergyLaw,
    SubunitExponent,
    SuperunitExponent,
    UnitExponent,
    energy_geometry,
)
from statadic.geometry.ground.config import (
    parse_ground_distance_name,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (  # noqa: TC001
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    CosineLaw,
    EuclideanLaw,
    GroundDistance,
    NormalizedEuclideanLaw,
    cdist,
    declare_ground_distance,
    euclidean_distance,
)


@dataclass(frozen=True, slots=True)
class CustomMetricLaw:
    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=LawProperties(0), init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)


def custom_metric(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)


def wrong_arity(x: jnp.ndarray) -> jnp.ndarray:
    return x


def needs_euclidean(distance: GroundDistance[EuclideanLaw]) -> None: ...
def needs_custom(distance: GroundDistance[CustomMetricLaw]) -> None: ...
def needs_separating(
    geometry: EnergyGeometry[
        EuclideanLaw,
        UnitExponent,
        SeparatingDistributionEnergyLaw,
    ],
) -> None: ...
def needs_nonseparating(
    geometry: EnergyGeometry[
        NormalizedEuclideanLaw,
        SuperunitExponent,
        DistributionEnergyLaw,
    ],
) -> None: ...
def needs_mean(
    geometry: EnergyGeometry[
        CosineLaw,
        UnitExponent,
        MeanEnergyLaw,
    ],
) -> None: ...


class ExactCallable:
    @property
    def law(self) -> EuclideanLaw:
        return EuclideanLaw()

    def __call__(self, left: int, right: str) -> float:
        return float(left + len(right))


class WrongResultCallable:
    @property
    def law(self) -> EuclideanLaw:
        return EuclideanLaw()

    def __call__(self, left: int, right: str) -> str:
        return str(left) + right


def invoke_exact(
    function: LawfulCallable[[int, str], float, EuclideanLaw],
) -> float:
    return function(1, "ok")


def accepted(points: jnp.ndarray) -> None:
    parsed = resolve_ground_distance(EUCLIDEAN)
    custom = declare_ground_distance(
        custom_metric,
        law=CustomMetricLaw(),
        name="custom",
        required_evidence=FiniteValues,
    )
    needs_euclidean(parsed)
    needs_custom(custom)
    cdist(points, points, metric=parsed)
    needs_separating(energy_geometry(EUCLIDEAN, UNIT_EXPONENT))
    needs_nonseparating(
        energy_geometry(NORMALIZED_EUCLIDEAN, SuperunitExponent(1.5))
    )
    needs_mean(energy_geometry(COSINE, UNIT_EXPONENT))
    invoke_exact(ExactCallable())
'''

_TYPED_DISTANCE_FIXTURE = (
    _TYPED_DISTANCE_ACCEPTED_FIXTURE
    + """\
def rejected(points: jnp.ndarray, dynamic_name: str) -> None:
    custom = declare_ground_distance(
        custom_metric,
        law=CustomMetricLaw(),
        name="custom",
        required_evidence=FiniteValues,
    )
    # serialized name in core
    cdist(points, points, metric="euclidean")  # E: is not assignable
    # bare callable
    cdist(points, points, metric=euclidean_distance)  # E: is not assignable
    # The host parsing boundary accepts a dynamic string and validates its value.
    parse_ground_distance_name(dynamic_name)
    declare_ground_distance(
        wrong_arity,  # E: is not assignable
        law=CustomMetricLaw(),
        required_evidence=FiniteValues,
    )  # ERROR: signature
    # angular CND witness
    energy_geometry(ANGULAR, SuperunitExponent(1.5))  # E: No matching overload
    # cosine CND witness
    energy_geometry(COSINE, SuperunitExponent(1.5))  # E: No matching overload
    # no ANOVA law
    energy_geometry(NORMALIZED_EUCLIDEAN, ANOVA_EXPONENT)  # E: No matching overload
    # exact custom law is retained
    needs_euclidean(custom)  # E: is not assignable
    quotient = energy_geometry(NORMALIZED_EUCLIDEAN, SuperunitExponent(1.5))
    needs_separating(quotient)  # E: is not assignable
    # ParamSpec/result are preserved
    invoke_exact(WrongResultCallable())  # E: is not assignable
"""
)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_axiom_lattice_is_a_static_error(tmp_path: Path) -> None:
    """Pyrefly rejects a widening ``DMat`` brand — enforced, not documented.

    Both directions matter. Invariance would also reject ``narrow()``, which is
    the legal direction. The independent clean fixture checks narrowing, while
    the expectation pins rejection to the widening call.
    """
    check_types(_WIDENING_ACCEPTED_FIXTURE, "widening_accepted.py", tmp_path)
    check_types(_WIDENING_FIXTURE, "widening.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_structural_law_capability_matrix_is_static(tmp_path: Path) -> None:
    """Pyrefly accepts every sound widening and rejects eight reverse flows."""
    check_types(
        _LAW_CAPABILITY_ACCEPTED_FIXTURE, "law_capabilities_accepted.py", tmp_path
    )
    check_types(_LAW_CAPABILITY_FIXTURE, "law_capabilities.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_mmd_brand_and_estimator_domains_are_static_errors(tmp_path: Path) -> None:
    """Pyrefly admits declared V-statistic MMD configurations only."""
    check_types(_MMD_BRAND_ACCEPTED_FIXTURE, "mmd_brands_accepted.py", tmp_path)
    check_types(_MMD_BRAND_FIXTURE, "mmd_brands.py", tmp_path)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_typed_distance_and_energy_regimes_are_static(tmp_path: Path) -> None:
    """Pyrefly preserves exact laws and rejects nine unsound boundary crossings."""
    check_types(
        _TYPED_DISTANCE_ACCEPTED_FIXTURE, "typed_distance_accepted.py", tmp_path
    )
    check_types(_TYPED_DISTANCE_FIXTURE, "typed_distance.py", tmp_path)


@pytest.mark.unit
@pytest.mark.contract
def test_static_alias_is_transparent() -> None:
    """``Static[T]`` erases to ``T`` and carries the marker in ``__metadata__``."""
    import typing  # introspection, not a runtime dependency  # noqa: PLC0415

    from statadic.core.typing import (  # noqa: PLC0415
        STATIC,
        Static,
    )

    # PEP 695 aliases are lazy: `Static[int]` records the substitution rather
    # than performing it, so the argument is read off the subscription and the
    # marker off the alias body.
    assert typing.get_args(Static[int]) == (int,)
    assert typing.get_args(Static.__value__)[0] is Static.__type_params__[0]
    assert STATIC in Static.__value__.__metadata__
